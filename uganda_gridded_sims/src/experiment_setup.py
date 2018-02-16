"""
Run multi-node simulation in Tororo region.
"""

import os
import numpy as np
import pandas as pd

from simtools.SetupParser import SetupParser

from dtk.tools.climate.ClimateGenerator import ClimateGenerator
from dtk.tools.migration.MigrationGenerator import MigrationGenerator
from dtk.tools.spatialworkflow.DemographicsGenerator import DemographicsGenerator
from dtk.utils.core.DTKConfigBuilder import DTKConfigBuilder
from dtk.vector.species import set_species_param
from dtk.interventions.itn_age_season import add_ITN_age_season
from dtk.interventions.health_seeking import add_health_seeking
from dtk.interventions.irs import add_IRS
from simtools.ExperimentManager.ExperimentManagerFactory import ExperimentManagerFactory
from simtools.ModBuilder import ModBuilder, ModFn
from simtools.SetupParser import SetupParser
from malaria.interventions.malaria_drug_campaigns import add_drug_campaign
from malaria.interventions.malaria_diagnostic import add_diagnostic_survey
# from dtk.utils.reports.MalariaReport import add_summary_report
from malaria.reports.MalariaReport import add_summary_report
from malaria.reports.MalariaReport import add_survey_report


from relative_time import *
from gridded_sim_general import *
from grid_ids_to_nodes import generate_lookup
from gen_migr_json import gen_gravity_links_json, save_link_rates_to_txt
from gridded_sim_general import *
from dtk.generic.properties import set_property
from dtk.generic.serialization import add_SerializationTimesteps
from dtk.generic.serialization import load_Serialized_Population
from malaria.interventions.malaria_drugs import drug_configs_from_code




class COMPS_spatial_experiment:

    def __init__(self,
                 base = os.getcwd(),
                 exp_name = None,
                 server = 'HPC',
                 catchment = 'all',
                 grid_pop_csv_fn = None,
                 immunity_fn = None,
                 immunity_mode = 'uniform',
                 larval_params_mode = 'uniform',
                 migration_on = True,
                 gravity_migration_params = None,
                 rcd_people_num = 5,
                 start_year = 2011,
                 sim_length_years = 10,
                 num_cores = 12,
                 healthseek_fn = None,
                 serialize_this_run = False,
                 serialize_day = 0,
                 run_from_pickup = False,
                 pickup_day = 0,
                 serialized_fp = None,
                 serialized_fn = None,
                 reporting_interval = 365,
                 reporting_trigger = 'Received_Cohort_Test',
                 healthseek_startday = 0,
                 input_fp = None,
                 reduce_nodes_to_just_enrolled = True
                 ):


        self.base = base
        self.exp_name = exp_name
        self.server = server
        self.exp_base = base + 'data/COMPS_experiment/{}/'.format(exp_name)
        self.input_base = base + 'input/'
        self.catchment = catchment
        self.grid_pop_csv_fn = grid_pop_csv_fn
        self.immunity_fn = immunity_fn
        self.immunity_mode = immunity_mode
        self.larval_params_mode = larval_params_mode
        self.migration_on = migration_on
        self.gravity_migration_params = gravity_migration_params
        self.rcd_people_num = rcd_people_num
        self.start_year = start_year
        self.sim_length_years = sim_length_years
        self.num_cores = num_cores
        self.healthseek_fn = healthseek_fn
        self.serialize_this_run = serialize_this_run
        self.serialize_day = serialize_day
        self.run_from_pickup = run_from_pickup
        self.pickup_day = pickup_day
        self.serialized_fp = serialized_fp
        self.serialized_fn = serialized_fn
        self.reporting_interval = reporting_interval
        self.reporting_trigger = reporting_trigger
        self.healthseek_startday = healthseek_startday
        self.input_fp = input_fp
        self.reduce_nodes_to_just_enrolled = reduce_nodes_to_just_enrolled


        self.ensure_filesystem()
        self.cb = self.build_cb()
        self.multinode_setup()
        self.basic_sim_seup()


    def ensure_filesystem(self):
        def ensure_dir(file_path):
            dir = os.path.dirname(file_path)
            if not os.path.exists(dir):
                os.makedirs(dir)
        ensure_dir(self.input_base)
        ensure_dir(self.exp_base)
        ensure_dir(self.exp_base + 'Demographics/')
        ensure_dir(self.exp_base + 'Immunity/')
        ensure_dir(self.exp_base + 'Climate/')
        ensure_dir(self.exp_base + 'Migration/')
        ensure_dir(self.exp_base + 'Logs/')




    def build_cb(self):


        SetupParser.default_block = self.server
        cb = DTKConfigBuilder.from_defaults('MALARIA_SIM')
        cb.set_experiment_executable(r'C:/Eradication/Eradication.exe')
        cb.set_input_files_root(self.exp_base)
        cb.set_dll_root(r'C:/Eradication/dll')

        return cb

    def basic_sim_setup(self):

        # Silencing StdOut error messaging

        self.cb.set_param("logLevel_SimulationEventContext", 'ERROR')
        self.cb.set_param("logLevel_VectorHabitat", 'ERROR')
        self.cb.set_param("logLevel_LarvalHabitatMultiplier", 'ERROR')

        #Set Vector params
        set_species_param(self.cb, 'gambiae', 'Larval_Habitat_Types',
                          {"TEMPORARY_RAINFALL":  1e9
                          })
        set_species_param(self.cb, "gambiae", "Indoor_Feeding_Fraction", 1.0)
        self.cb.update_params({"x_Temporary_Larval_Habitat": 1})
        # Miscellaneous

        self.cb.set_param("Enable_Demographics_Other", 1)
        self.cb.set_param("Enable_Demographics_Builtin", 0)
        self.cb.set_param("Valid_Intervention_States", [])
        self.cb.set_param("New_Diagnostic_Sensitivity", 0.025) # 40/uL
        self.cb.set_param("Simulation_Duration", 365*self.sim_length_years)

        # Report Events Recorder for not standard reporting
        self.full_events_list = ["Bednet_Got_New_One",
                                 "Bednet_Using",
                                 "Bednet_Discarded",
                                 "Received_Treatment",
                                 "Received_Campaign_Drugs",
                                 "Received_Test",
                                 "Received_Cohort_Test",
                                 "TestedPositive",
                                 "Received_RCD_Drugs"]

        self.cb.update_params({
            "Report_Event_Recorder": 1,
            "Report_Event_Recorder_Ignore_Events_In_List": 0,
            "Report_Event_Recorder_Events": self.full_events_list
        })

        self.cb.update_params({
            'Enable_Spatial_Output': 0,
            'Spatial_Output_Channels': ['Infectious Vectors',
                                        'Adult_Vectors',
                                        'New_Infections',
                                        'Population',
                                        'Prevalence',
                                        'New_Diagnostic_Prevalence',
                                        'Daily_EIR',
                                        'New_Clinical_Cases',
                                        'Human_Infectious_Reservoir',
                                        'Daily_Bites_Per_Human',
                                        'Land_Temperature',
                                        'Relative_Humidity',
                                        'Rainfall',
                                        'Air_Temperature']
        })

        add_summary_report(self.cb)
        add_survey_report(self.cb, reporting_interval= self.reporting_interval, trigger = self.reporting_trigger)

        self.implement_healthseeking()


    def implement_healthseeking(self):

        add_health_seeking(self.cb,
                           start_day=self.healthseek_startday,
                           targets = [{'trigger': 'NewClinicalCase',
                                       'coverage': 1,
                                       'agemin':0,
                                       'agemax':5,
                                       'seek':1,
                                       'rate': 0.3
                                       },
                                      {'trigger': 'NewClinicalCase',
                                       'coverage': 1,
                                       'agemin': 5,
                                       'agemax': 200,
                                       'seek': 1,
                                       'rate': 0.3
                                       },
                                      {'trigger': 'NewSevereCase',
                                       'coverage': 1,
                                       'agemin': 0,
                                       'agemax': 5,
                                       'seek': 1,
                                       'rate': 0.3
                                       },
                                      {'trigger': 'NewSevereCase',
                                       'coverage': 1,
                                       'agemin': 5,
                                       'agemax': 200,
                                       'seek': 1,
                                       'rate': 0.3
                                       }]

                           )


    def multinode_setup(self):
        self.demo_fp = 'Demogaphics/demo.json'
        self.demo_fp_full = os.path.join(self.exp_base,self.demo_fp)
        self.immun_fp = 'Immunity/immun.json'
        self.immun_fp_full = os.path.join(self.exp_base,self.immun_fp)
        self.enroll_fp = 'Demographics/enroll.json'
        self.enroll_fp_full = os.path.join(self.exp_base,self.enroll_fp)

        self.cb.set_param("Num_cores", self.num_cores)
        self.cb.update_params({'Demographics_Filenames': [self.demo_fp, self.immun_fp, self.enroll_fp]})
        #######################################################################################################
        # CLIMATE-RELATED PARAMETERS:
        #######################################################################################################
        self.cb.update_params({
            'Air_Temperature_Filename': "Uganda_30arcsec_air_temperature_daily.bin",
            'Land_Temperature_Filename': "Uganda_30arcsec_air_temperature_daily.bin",
            'Rainfall_Filename': "Uganda_30arcsec_rainfall_daily.bin",
            'Relative_Humidity_Filename': "Uganda_30arcsec_relative_humidity_daily.bin"
        })

        #######################################################################################################
        # MIGRATION-RELATED PARAMETERS:
        #######################################################################################################
        if self.migration_on:
            self.cb.update_params({
                                    'Migration_Model': 'FIXED_RATE_MIGRATION',
                                    'Local_Migration_Filename': 'local_migration.bin', # note that underscore prior 'migration.bin' is required for legacy reasons that need to be refactored...
                                    'Enable_Local_Migration':1,
                                    'Migration_Pattern': 'SINGLE_ROUND_TRIPS', # human migration
                                    'Local_Migration_Roundtrip_Duration': 2, # mean of exponential days-at-destination distribution
                                    'Local_Migration_Roundtrip_Probability': 0.95, # fraction that return
                                    'x_Local_Migration': 4 # amplitude of migration
            })
        elif not self.migration_on:
            self.cb.update_params({'Migration_Model': 'NO_MIGRATION'})  #'NO_MIGRATION' is actually default for MALARIA_SIM, but might as well make sure it's off

    def update_nodes_in_demo(self, fractions_by_node):

        with open(self.demo_fp_full, 'r') as f:
            demo_dict = json.load(f)
            n_nodes = len(demo_dict["Nodes"])
            d = {}
            d["Nodes"] = []
            d["NodeCount"] = n_nodes

            def define_enrollment_property_per_node(frac_enrolled):
                prop = set_property('Enrolled', ['Yes', 'No'], [frac_enrolled, 1 - frac_enrolled])
                return [prop]

            for node in demo_dict['Nodes']:
                fac_name = int(node['NodeAttributes']['FacilityName'])

                if fac_name in fractions_by_node:  # update as a dictionary of keys value pairs

                    frac_enrolled = fractions_by_node[fac_name]
                    node = {'NodeID': node['NodeID']}
                    node['IndividualProperties'] = define_enrollment_property_per_node(frac_enrolled=frac_enrolled)
                    d['Nodes'].append(node)
                else:
                    node = {'NodeID': node['NodeID']}
                    node['IndividualProperties'] = define_enrollment_property_per_node(frac_enrolled=0)

                    if self.reduce_nodes_to_just_enrolled:
                        node['NodeAttributes'] = dict(InitialPopulation=0)
                    d['Nodes'].append(node)

            with open(self.enroll_fp_full, 'w') as f:
                json.dump(d,f,indent=4)

    def gen_demo_file(self,input_csv,larval_params=None):
        dg = DemographicsGenerator.from_file(self.cb, input_csv, catch=self.catch)
        demo_dict = dg.generate_demographics()

        # Add larval habitat parameters to demographics file:
        demo_dict = self.add_larval_habitats_to_demo(demo_dict, larval_params=larval_params)

        if larval_params:
            temp_h = larval_params['temp_h']
            linear_h = larval_params['linear_h']
            demo_fp = self.exp_base + "Demographics/demo_temp{}_linear{}.json".format(int(temp_h),int(linear_h))
        else:
            demo_fp = self.exp_base + "Demographics/demo.json"

        demo_f = open(demo_fp, 'w+')
        json.dump(demo_dict, demo_f, indent=4)
        demo_f.close()

    #################################################################################################
    # LARVAL PARAMETER-RELATED FUNCTIONS:
    def add_larval_habitats_to_demo(self, demo_dict, nodeset='all', larval_params=None):
        # Add larval habitat parameters (some of which scale with population) to demographics file

        def add_larval_habitat_to_node(node_item, const_h, temp_h, water_h, linear_h):
            calib_single_node_pop = 1000  # for Zambia

            # This is now done in the demographics generator itself:
            # birth_rate = (float(node_item['NodeAttributes']['InitialPopulation']) / (1000 + 0.0)) * 0.12329
            # node_item['NodeAttributes']['BirthRate'] = birth_rate

            pop_multiplier = float(node_item['NodeAttributes']['InitialPopulation']) / (calib_single_node_pop + 0.0)

            temp_multiplier = temp_h * pop_multiplier
            linear_multiplier = linear_h * pop_multiplier
            const_multiplier = const_h  # NOTE: No pop multiplier
            water_multiplier = water_h * pop_multiplier

            node_item['NodeAttributes']['LarvalHabitatMultiplier'] = {
                "CONSTANT": const_multiplier,
                "TEMPORARY_RAINFALL": temp_multiplier,
                "WATER_VEGETATION": water_multiplier,
                "LINEAR_SPLINE": linear_multiplier
            }

        # if self.larval_params_mode == 'milen':
        #     # get grid cells from pop csv file:
        #     # for those grid cells, get corresponding arab/funest params
        #     # loop over nodes [order will correspond, by construction, to pop csv ordering]
        #     # give each node the corresponding larval params
        #
        #     # Load pop csv file to get grid cell numbers:
        #     # pop_df = pd.read_csv(self.grid_pop_csv_file)
        #     # grid_cells = np.array(pop_df['node_label'])
        #     grid_cells = find_cells_for_this_catchment(self.catch)
        #
        #     # From those grid cells, and the Milen-clusters they correspond to, get best-fit larval habitat parameters
        #     arab_params, funest_params = find_milen_larval_param_fit_for_grid_cells(grid_cells,
        #                                                                             fudge_milen_habitats=self.fudge_milen_habitats)
        #
        #     # Loop over nodes in demographics file (which will, by construction, correspond to the grid pop csv ordering)
        #     i = 0
        #     for node_item in demo_dict['Nodes']:
        #         # if larval_params:
        #         #     const_h = larval_params['const_h']
        #         #     temp_h = larval_params['temp_h']
        #         #     water_h = larval_params['water_h']
        #         #     linear_h = larval_params['linear_h']
        #         # else:
        #         const_h = 1.
        #         temp_h = arab_params[i]
        #         water_h = 1.
        #         linear_h = funest_params[i]
        #
        #         add_larval_habitat_to_node(node_item, const_h, temp_h, water_h, linear_h)
        #
        #         i += 1

        if self.larval_params_mode == 'uniform':
            for node_item in demo_dict['Nodes']:

                if larval_params:
                    const_h = larval_params['const_h']
                    temp_h = larval_params['temp_h']
                    water_h = larval_params['water_h']
                    linear_h = larval_params['linear_h']
                else:
                    const_h = 1.
                    temp_h = 122.
                    water_h = 1.
                    linear_h = 97.

                add_larval_habitat_to_node(node_item, const_h, temp_h, water_h, linear_h)

        return demo_dict

    def larval_param_sweeper(self, cb, temp_h, linear_h):
        larval_params = {
            "const_h": 1.,
            "water_h": 1.,
            "temp_h": np.float(temp_h),
            "linear_h": np.float(linear_h)
        }

        # Will need new demographics file that incorporates these larval parameters.
        # demographics filename example: Demographics/MultiNode/demo_temp50_linear120.json
        new_demo_fp = self.demo_fp[:-5] + "_temp{}_linear{}.json".format(temp_h, linear_h)

        # Check if demographics file for these parameters already exists.  If not, create it
        if not os.path.isfile(new_demo_fp):
            self.gen_demo_file(self.grid_pop_csv_file, larval_params=larval_params)

        # Then pass this demographics file to the config_builder.
        if self.immunity_on:
            self.cb.update_params({'Demographics_Filenames': [new_demo_fp, self.immun_fp]})
        else:
            self.cb.update_params({'Demographics_Filenames': [new_demo_fp]})

        return larval_params

        #################################################################################################

    # MIGRATION-RELATED FUNCTIONS:
    def gen_migration_files(self):
        migr_json_fp = self.exp_base + "Migration/grav_migr_rates.json"

        migr_dict = gen_gravity_links_json(self.demo_fp_full, self.gravity_migr_params, outf=migr_json_fp)
        rates_txt_fp = self.exp_base + "Migration/grav_migr_rates.txt"

        save_link_rates_to_txt(rates_txt_fp, migr_dict)

        # Generate migration binary:
        migration_filename = self.cb.get_param('Local_Migration_Filename')
        print("migration_filename: ", migration_filename)
        MigrationGenerator.link_rates_txt_2_bin(rates_txt_fp,
                                                self.exp_base + migration_filename)

        # Generate migration header:
        MigrationGenerator.save_migration_header(self.demo_fp_full,
                                                 self.exp_base + 'Migration/local_migration.bin.json'
                                                 )

    def vector_migration_sweeper(self, vector_migration_on):
        if vector_migration_on:
            self.cb.update_params({
                'Vector_Migration_Modifier_Equation': 'LINEAR',
                'Vector_Sampling_Type': 'SAMPLE_IND_VECTORS',  # individual vector model (required for vector migration)
                'Mosquito_Weight': 10,
                'Enable_Vector_Migration': 1,  # mosquito migration
                'Enable_Vector_Migration_Local': 1,
            # migration rate hard-coded in NodeVector::processEmigratingVectors() such that 50% total leave a 1km x 1km square per day (evenly distributed among the eight adjacent grid cells).
                'Vector_Migration_Base_Rate': 0.15,  # default is 0.5
                'x_Vector_Migration_Local': 1
            })
        else:
            self.cb.update_params({
                'Enable_Vector_Migration': 0,  # mosquito migration
                'Enable_Vector_Migration_Local': 0
                # migration rate hard-coded in NodeVector::processEmigratingVectors() such that 50% total leave a 1km x 1km square per day (evenly distributed among the eight adjacent grid cells).
            })
        return {"vec_migr": vector_migration_on}

        #################################################################################################


    def implement_interventions(self, cb, broadcast_msat, broadcast_dummy_to_enrolled):
        start_date = "{}-01-01".format(self.start_year)  # Day 1 of simulation
        date_format = "%Y-%m-%d"
        sim_duration = 365 * self.sim_length_years  # length in days
        self.cb.params['Simulation_Duration'] = sim_duration

        # Prevent DTK from spitting out too many messages
        self.cb.params['logLevel_JsonConfigurable'] = "WARNING"
        self.cb.params['Disable_IP_Whitelist'] = 1

        # Get grid cell to node ID lookup table:
        from grid_ids_to_nodes import generate_lookup
        nodeid_lookup, pop_lookup = generate_lookup(self.demo_fp_full)


        def add_cohort_test(cb, start_days, coverage, drug_configs, repetitions, interval,
                            diagnostic_type, diagnostic_threshold, node_cfg
                            ):

            for start_day in start_days:
                add_diagnostic_survey(cb, coverage=coverage, repetitions=repetitions, tsteps_btwn=interval,
                                      start_day=start_day,
                                      received_test_event='Received_Cohort_Test',
                                      diagnostic_type=diagnostic_type, diagnostic_threshold=diagnostic_threshold,
                                      node_cfg=node_cfg, positive_diagnosis_configs=drug_configs,

                                      )
        if broadcast_msat:
            add_cohort_test(cb, start_days=[self.serialize_day], drug_configs=drug_configs_from_code(cb, 'AL'),
                            coverage=1, repetitions=18, interval=30, diagnostic_type='TRUE_PARASITE_DENSITY',
                            diagnostic_threshold=40, node_cfg={"class": "NodeSetAll"}
                            )

        if broadcast_dummy_to_enrolled:
            add_drug_campaign(cb, campaign_type='MDA', drug_code='Vehicle', start_days=[self.serialize_day], coverage=1,
                              repetitions=18,
                              interval=30,
                              ind_property_restrictions=[{'Enrolled': 'Yes'}])


        return {
                'broadcastMSAT': broadcast_msat,
                'broadcast_dummy_to_enrolled': broadcast_dummy_to_enrolled}


    def file_setup(self,generate_demographics_file=True,generate_climate_files=True,generate_migration_files=True,generate_enrollment_files = True):
        # self.multinode_setup()

        if generate_demographics_file:
            print("Generating demographics file...")
            self.gen_demo_file(self.grid_pop_csv_file)

        if generate_migration_files:
            print("Generating migration files...")
            self.gen_migration_files()

        if generate_climate_files:
            print("Generating climate files...")
            SetupParser.init()
            cg = ClimateGenerator(self.demo_fp_full,
                                  self.exp_base + 'Logs/climate_wo.json',
                                  self.exp_base + 'Climate/',
                                  start_year = str(self.start_year),
                                  num_years = str(np.min([2016 - self.start_year, self.sim_length_years])))
            cg.generate_climate_files()
        if generate_enrollment_files:
            print('Generating enrollment files...')
            self.update_nodes_in_demo(fractions_by_node={4679: 10/330 , 4680: 5/73})





    def submit_experiment(self,num_seeds=1,intervention_sweep=False,larval_sweep=False,migration_sweep=False,vector_migration_sweep=False,
                          simple_intervention_sweep=True,custom_name=None):

        # Implement the actual (not dummy) baseline healthseeking
        self.implement_baseline_healthseeking()


        modlists = []

        if num_seeds > 1:
            new_modlist = [ModFn(DTKConfigBuilder.set_param, 'Run_Number', seed) for seed in range(num_seeds)]
            modlists.append(new_modlist)

        if larval_sweep:
            new_modlist = [ModFn(self.larval_param_sweeper, temp_h, linear_h)
                           for temp_h in [61,122,244]
                           for linear_h in [48, 97, 194]]
            modlists.append(new_modlist)

        if migration_sweep:
            new_modlist = [ModFn(DTKConfigBuilder.set_param, 'x_Local_Migration', x) for x in [0.5,1,2,5]]
            modlists.append(new_modlist)

        if vector_migration_sweep:
            new_modlist = [ModFn(self.vector_migration_sweeper, vector_migration_on) for vector_migration_on in [True, False]]
            modlists.append(new_modlist)

        if simple_intervention_sweep:
            new_modlist = [
                ModFn(self.implement_interventions, True, False, False, False, False),
                ModFn(self.implement_interventions, False, True, False, False, False),
                ModFn(self.implement_interventions, False, False, True, False, False),
                ModFn(self.implement_interventions, False, False, False, True, False),
                ModFn(self.implement_interventions, False, False, False, False, True),
                ModFn(self.implement_interventions, True, True, True, True, True)
            ]
            modlists.append(new_modlist)
        else:
            new_modlist = [ModFn(self.implement_interventions, True, True, True, True,True,True,True)]
            modlists.append(new_modlist)



        # if intervention_sweep:
        #     # Interventions to turn on or off
        #     include_itn_list = [True, False]
        #     include_irs_list = [True, False]
        #     include_mda_list = [True, False]
        #     include_msat_list = [True, False]
        #     include_stepd_list = [True, False]
        #
        #     new_modlist = [
        #         ModFn(self.implement_interventions, use_itn, use_irs, use_msat, use_mda, use_stepd)
        #         for use_itn in include_itn_list
        #         for use_irs in include_irs_list
        #         for use_mda in include_mda_list
        #         for use_msat in include_msat_list
        #         for use_stepd in include_stepd_list
        #     ]
        #
        # else:
        #     new_modlist = [ModFn(self.implement_interventions, True, True, True, True, True)]
        # modlists.append(new_modlist)


        builder = ModBuilder.from_combos(*modlists)

        run_name = self.exp_name
        if custom_name:
            run_name = custom_name

# Module for serialization

        if self.serialize_this_run:
            add_SerializationTimesteps(self.cb, [self.serialize_day], end_at_final=True)

        if self.run_from_pickup:
            load_Serialized_Population(self.cb, self.serialized_filepath, self.serialized_fn)
            self.cb.update_params({'Start_Time': self.pickup_day})

        # SetupParser.init()
        # SetupParser.set("HPC","priority","Normal")
        exp_manager = ExperimentManagerFactory.init()
        exp_manager.run_simulations(config_builder=self.cb, exp_name=run_name, exp_builder=builder)

