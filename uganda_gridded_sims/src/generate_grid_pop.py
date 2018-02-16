import os
import sys
import json
import math
import unicodedata
import logging
from math import pow, radians, cos, sin, asin, sqrt


import numpy as np
import numpy.ma as ma

import pandas as pd
import matplotlib as mpl

from shapely.geometry import shape, Point
from descartes import PolygonPatch
from pyproj import Proj, transform
from affine import Affine

import rasterio
import rasterio.drivers
import rasterio.mask

import matplotlib.pyplot as plt



def get_haversine_distance(lon1, lat1, lon2, lat2):
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))

    # 6367 km is the radius of the Earth
    km = 6367 * c

    return km


"""
get the extent (bbox) of single shape's features
"""


def get_shape_bbox(shape):
    def explode(coords):

        for e in coords:
            if isinstance(e, (float, int, long)):
                yield coords
                break
            else:
                for f in explode(e):
                    yield f

    x, y = zip(*list(explode(shape['geometry']['coordinates'])))
    return min(x), max(x), min(y), max(y)


def remove_accents(s):
    if type(s) == 'str':
        s = unicode(s, 'utf-8')
    return ''.join(x for x in unicodedata.normalize('NFKD', s)
                   if unicodedata.category(x) != 'Mn').lower().replace('-', ' ')


def load_geojson_shapes(shapes, filter_shapes=None):
    shape_records = {}

    x_min = 1000
    x_max = -1000
    y_min = 1000
    y_max = -1000

    for shape_features in shapes["features"]:

        shape_name = remove_accents(shape_features["properties"]["admin1Name"])

        if not filter_shapes:
            shape_records[shape_name] = shape_features
        elif shape_name in filter_shapes:
            shape_records[shape_name] = shape_features
        else:
            continue

        x_mint, x_maxt, y_mint, y_maxt = get_shape_bbox(shape_features)

        x_min = min(x_mint, x_min)
        x_max = max(x_maxt, x_max)
        y_min = min(y_mint, y_min)
        y_max = max(y_maxt, y_max)

    extent = [x_min, x_max, y_min, y_max]

    return shape_records, extent


def reproj_coords(x, y):
    rx, ry = transform(
        Proj(init='epsg:32632'),
        Proj(init='epsg:4326'),
        x, y
    )

    return rx, ry


def shape_filter(point, shapes):
    # decide if a point is in a geojson shape
    def is_in_shape(point, shape):
        return shape.contains(Point(float(point["lon"]), float(point["lat"])))

    for shape_features in shapes:
        shape_geometry = shape(shape_features["geometry"])

        if is_in_shape(point, shape_geometry):
            return True

    return False


def get_raster_value_at_point(gt, pos):
    x = int((pos[0] - gt[0]) / gt[1])
    y = int((pos[1] - gt[3]) / gt[5])

    return data[y, x]


#import the csv containing latlongs
latlong_fp = r'C:\Uganda\all_data_w_lat_long.csv'

df_latlongs = pd.read_csv(latlong_fp,usecols=['siteid','hhid','Longitude','Latitude'])
df_latlongs = df_latlongs[df_latlongs.siteid == 'Tororo']
df_latlongs.drop_duplicates()
df_latlongs.to_csv(r'C:\Users\jorussell\PycharmProjects\SpatialTororo\uganda_gridded_sims\data\Tororo_unique_household_latlongs.csv')


x_min = min(df_latlongs.Longitude)
x_max = max(df_latlongs.Longitude)
y_min = min(df_latlongs.Latitude)
y_max = max(df_latlongs.Latitude)

print(x_min,x_max,y_min,y_max)

#import the raster data
data_directory = r'C:\Users\jorussell\PycharmProjects\SpatialTororo\uganda_gridded_sims\data'
raster_fp = r'C:\Users\jorussell\Dropbox (IDM)\Malaria Team Folder\data\Uganda\PRISM\hrsl_uga_v1\uga\hrsl_uga.tif'
tororo_geojson_fp = r'C:\Users\jorussell\PycharmProjects\SpatialTororo\uganda_gridded_sims\data\tororo_shapefile.geojson'

with open(tororo_geojson_fp) as f:
    tororo_shp = json.load(f)


pop = rasterio.open(raster_fp, "r")

masked_raster, out_transform = rasterio.mask.mask(pop, [tororo_shp["features"][0]['geometry']], crop = True)
masked_raster = masked_raster[0] # extracting 2D raster (mask returns a 3d ndarray)
masked_meta = pop.meta.copy()
masked_meta.update({
                        "height": masked_raster.shape[0],
                        "width": masked_raster.shape[1],
                        "transform": out_transform})


clipped_dataset = rasterio.open(os.path.join(data_directory, "clipped.tif"), "w", **masked_meta)
clipped_dataset.write(masked_raster, 1)
clipped_dataset.close()

with rasterio.open(os.path.join(data_directory, "clipped.tif")) as r:
    T0 = r.affine  # upper-left pixel corner affine transform
    p1 = Proj(r.crs)
    A = r.read(1)  # pixel values

    # All rows and columns
    cols, rows = np.meshgrid(np.arange(A.shape[1]), np.arange(A.shape[0]))

    # Get affine transform for pixel centres
    T1 = T0 * Affine.translation(0.5, 0.5)

    # Function to convert pixel row/column index (from 0) to easting/northing at centre
    rc2en = lambda r, c: (c, r) * T1

    # All eastings and northings (there is probably a faster way to do this)
    eastings, northings = np.vectorize(rc2en, otypes=[np.float, np.float])(rows, cols)

    # Project all longitudes, latitudes
    p2 = Proj(proj='latlong', datum='WGS84')
    lats, longs = transform(p1, p2, eastings, northings)

    pop_threshold = 2
    cells_masking = np.ma.masked_less(A, pop_threshold)
    cells_mask = cells_masking.mask
    cells_data = cells_masking.data

    # mask returns False for valid entries;  True would be easier to work with
    inverted_filtered_mask = np.in1d(cells_mask.ravel(), [False]).reshape(cells_mask.shape)
    filtered_idx = np.where(inverted_filtered_mask)

    grid_out = "lon,lat,value\n"
    for i, idx_x in enumerate(filtered_idx[0]):
        idx_y = filtered_idx[1][i]

        lat = str(lats[idx_x][idx_y])
        lon = str(longs[idx_x][idx_y])
        popn = str(int(cells_data[idx_x][idx_y]))

        grid_out += lat + ','
        grid_out += lon + ','
        grid_out += popn + "\n"

with open(os.path.join(data_directory, "clipped.csv"), "w") as o_f:
    o_f.write(grid_out)

# dataset = gdal.Open(raster_fp, gdal.GA_ReadOnly)
# print("Driver: {}/{}".format(dataset.GetDriver().ShortName,
#                              dataset.GetDriver().LongName))
# print("Size is {} x {} x {}".format(dataset.RasterXSize,
#                                     dataset.RasterYSize,
#                                     dataset.RasterCount))
# print("Projection is {}".format(dataset.GetProjection()))
# geotransform = dataset.GetGeoTransform()
# if geotransform:
#     print("Origin = ({}, {})".format(geotransform[0], geotransform[3]))
#     print("Pixel Size = ({}, {})".format(geotransform[1], geotransform[5]))
#
#
# band = dataset.GetRasterBand(1)
# print("Band Type={}".format(gdal.GetDataTypeName(band.DataType)))
#
# min_ = band.GetMinimum()
# max_ = band.GetMaximum()
# if not min or not max:
#     (min_, max_) = band.ComputeRasterMinMax(True)
# print("Min={:.3f}, Max={:.3f}".format(min_, max_))
#
# if band.GetOverviewCount() > 0:
#     print("Band has {} overviews".format(band.GetOverviewCount()))
#
# if band.GetRasterColorTable():
#     print("Band has a color table with {} entries".format(band.GetRasterColorTable().GetCount()))
#
# scanline = band.ReadRaster(xoff=0, yoff=0,
#                            xsize=band.XSize, ysize=1,
#                            buf_xsize=band.XSize, buf_ysize=1,
#                            buf_type=gdal.GDT_Float32)
#
# test_band = band.ReadAsArray()
#
#
# nodata = band.GetNoDataValue()
#
# import struct
# tuple_of_floats = struct.unpack('f'*band.XSize, scanline)

# square grid cell/pixel size
cell_size = 1000

#average household size (in people)
avg_household_size = 4.5

#define area around Tororo
