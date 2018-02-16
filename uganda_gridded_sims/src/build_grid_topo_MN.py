'''
- process a collection of households into a connected grid of cells
- input:     - households csv file with required columns lat,lon # see example input files (structures_households.csv)
             - optional: using geojson shape based household/grid cell filter requires
- output:    - csv file of grid locations # see example output files (pop_gridded.csv)
             - json adjacency list of local grid connectivity (compatible with dtk-tools spatial workflow) # see example output files (gridded_households_adj_list.json)
'''

import math
import json
import logging
import unicodedata

import numpy as np
import numpy.ma as ma

import pandas as pd

from geopy.distance import vincenty
import geocoder

from sklearn.neighbors import DistanceMetric

from sklearn.neighbors import NearestNeighbors

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.collections import PatchCollection

# square grid cell/pixel side (in m)
cell_size = 500

# demographic grid cell should contain more households than a threshold
cell_household_threshold = 1

# how far people would definitely go by foot in units of neighborhood hops (1 hop is the adjacent 8 cells on the grid; 2 hops is the adjacent 24 cells, etc.
# this prepares an approximation of a local topology
migration_radius = 3

# average household size (in people); verify with census/other data?
avg_household_size = 4.5  # can change to distribution...

# lat lon bounding box to filter input buildings if needed


# area around Moron

x_min = 33.970767
x_max = 34.082351
y_min = 0.7146171
y_max = 0.805583

'''
# Grand'Anse
# if you test with the below bounding box you may run into NaN values in climate files causing errors when DTK is run
# this is a pretty big area and you might want to use 500m cell size above
# ping Milen if you run into that or have any questions about climate
x_min = -74.544151
x_max = -73.784895
y_min = 18.362657
y_max = 18.699085
'''

if __name__ == '__main__':

    logging.basicConfig(format='%(message)s', level='INFO')

    logging.info("Reading data...")

    # get household records within the given bounding box
    all_hh_records = pd.read_csv(r'C:\users\jorussell\PycharmProjects\SpatialTororo\uganda_gridded_sims\data\clipped.csv')
    hh_records = all_hh_records[
        (all_hh_records.lon > x_min) & (all_hh_records.lon < x_max) & (all_hh_records.lat > y_min) & (
        all_hh_records.lat < y_max)]

    # get point locations of households
    points = hh_records.as_matrix(["lon", "lat"])

    logging.info("Calculating grid cell mesh...")

    # get number of grid cells along the x (grid width) and y axis (grid height) based on the bounding box dimensions and pixel/cell size
    num_cells_x = int(1000 * vincenty((y_min, x_min), (y_min, x_max)).km / cell_size) + 1
    num_cells_y = int(1000 * vincenty((y_min, x_min), (y_max, x_min)).km / cell_size) + 1

    # bin households in the grid
    H, xedges, yedges = np.histogram2d(points[:, 0], points[:, 1], bins=[num_cells_x, num_cells_y])

    # get centroids of grid cells
    x_mid = (xedges[1:] + xedges[:-1]) / 2
    y_mid = (yedges[1:] + yedges[:-1]) / 2

    # build a mesh of centroids
    X_mid, Y_mid = np.meshgrid(x_mid[:], y_mid[:])

    # build a mesh of grid cell vertexes
    X, Y = np.meshgrid(xedges[:-1], yedges[:-1])

    # plot grid cells; color by density
    fig = plt.figure(figsize=(13, 9))

    logging.info("Plotting all filtered grid cells...")

    plt.pcolormesh(X, Y, np.swapaxes(H, 0, 1), cmap='coolwarm', vmin=0, vmax=20)
    plt.colorbar()
    plt.show()

    # filter pixels/cells by number of households greater than a threshold in each cell
    cells_masking = ma.masked_less(H, cell_household_threshold)
    cells_mask = cells_masking.mask
    cells_data = cells_masking.data

    # mask returns False for valid entries;  True would be easier to work with
    inverted_filtered_households_mask = np.in1d(cells_mask.ravel(), [False]).reshape(cells_mask.shape)
    filtered_household_cells_idx = np.where(inverted_filtered_households_mask)

    logging.info("Constructing population nodes...")

    # get coordinates of filtered cells' midpoints and calculate approximate population per pixels using number of households and avg # people (guess) per household
    # use that to assemble population nodes
    pop_nodes = "node_label,lat,lon,pop,num_hhs\n"

    # map between node coordinates and node label; to avoid look-up logic
    coor_idxs_2_node_label = {}

    for i, idx_x in enumerate(filtered_household_cells_idx[0]):
        idx_y = filtered_household_cells_idx[1][i]

        node_label = str(i)  # unique node label

        coor_idxs_2_node_label[str(idx_x) + "_" + str(idx_y)] = node_label

        lat = str(Y_mid[idx_y][idx_x])
        lon = str(X_mid[idx_y][idx_x])
        pop = str(int(cells_data[idx_x][idx_y] * avg_household_size))
        num_hhs = str(cells_data[idx_x][idx_y])

        pop_nodes += node_label + "," + lat + "," + lon + "," + pop + "," + num_hhs + "\n"

    with open("pop_gridded.csv", "w") as phg_f:
        phg_f.write(pop_nodes)

    logging.info("Saved grid population nodes to pop_gridded.csv")

    logging.info("Generating grid graph adjacency matrix")

    '''
    Construct the grid graph adjacency list based on local connectivity constraints

    There are more efficient ways to do that, but for graphs of dozens of thousands of nodes this is probably fine
    '''
    adj_list = {}

    # get each cell in the grid
    for i, idx_x in enumerate(filtered_household_cells_idx[0]):
        idx_y = filtered_household_cells_idx[1][i]
        node_label = str(i)

        adj_list[node_label] = {}

        # find the potential neighbors of the cell
        neigh_candidates = []
        for i in range(idx_x - migration_radius, idx_x + migration_radius + 1):
            for j in range(idx_y - migration_radius, idx_y + migration_radius + 1):
                neigh_candidates.append([i, j])

        # check if neighbor cells exist on the grid (e.g. their household density is sufficiently high)
        for neigh_cand in neigh_candidates:
            if str(neigh_cand[0]) + "_" + str(neigh_cand[1]) in coor_idxs_2_node_label:
                neigh_node_label = coor_idxs_2_node_label[str(neigh_cand[0]) + "_" + str(neigh_cand[1])]

                lat_src = Y_mid[idx_y][idx_x]
                lon_src = X_mid[idx_y][idx_x]

                lat_dst = Y_mid[neigh_cand[1]][neigh_cand[0]]
                lon_dst = X_mid[neigh_cand[1]][neigh_cand[0]]

                # calculate source destination distance and weight the graph
                w = vincenty((lat_src, lon_src), (lat_dst, lon_dst)).km

                adj_list[node_label][neigh_node_label] = w

    with open("gridded_households_adj_list.json", "w") as a_f:
        json.dump(adj_list, a_f, indent=3)

    logging.info("Grid graph adjacency matrix saved to gridded_households_adj_list.json")

    # plotting grid cells
    logging.info("Plotting box-filtered grid cells.")
    nodes_records = pd.read_csv("pop_gridded.csv")
    points = nodes_records.as_matrix(["lon", "lat", "pop"])
    plt.scatter(points[:, 0], points[:, 1], s=np.sqrt(points[:, 2]), c="red", linewidths=0)

    plt.show()