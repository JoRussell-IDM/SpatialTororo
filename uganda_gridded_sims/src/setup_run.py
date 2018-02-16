from experiment_setup import *
from simtools.SetupParser import SetupParser
import sys

base = 'C:/Users/jorussell/PycharmProjects/SpatialTororo/uganda_gridded_sims/'
grid_pop_csv_file = base + 'data/pop_gridded.csv' # Max pop in grid cell for entire region


comps_exp = COMPS_Experiment(base,
                             exp_name,
                             catch=catch,
                             grid_pop_csv_file=grid_pop_csv_file,
                             imm_1node_fp=imm_1node_fp,
                             migration_on=True,
                             start_year=2007,
                             sim_length_years=num_years,
                             rcd_people_num=10,
                             gravity_migr_params=gravity_migr_params,
                             num_cores=num_cores,
                             healthseek_fn=healthseek_fn,
                             itn_fn=itn_fn,
                             irs_fn=irs_fn,
                             msat_fn=msat_fn,
                             mda_fn=mda_fn,
                             stepd_fn=stepd_fn,
                             larval_params_mode="uniform",
                             immunity_mode="uniform",
                             serialize_this_run = False,
                             serialize_day = serialize_day,
                             pickup_day = serialize_day,
                             run_from_pickup=True,
                             serialized_filepath= serialized_filepath,
                             serialized_fn = serialized_filenames)