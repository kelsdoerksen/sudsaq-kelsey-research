"""
Script for processing the GEE features to be array format
to use in n-channel image input for UNet
"""

import numpy as np
import xarray as xr
from scipy import stats
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Preprocessing GEE data')
    parser.add_argument('--region', help='Location, na, eu or globe')
    parser.add_argument('--gee_dir', help='Directory of gee data to process')
    return parser.parse_args()


"""
Pseudo-code:
1. Read in gee data
2. Subset over region of interest
3. Save
"""


def max_min_normalize_array(array):
  '''
  Normalize the array so that it's between 0-1
  max min normaliztion = x-min/max-min for each x in array
  '''
  arr_min = np.amin(array)
  arr_max = np.amax(array)

  norm_array = (array - arr_min)/(arr_max-arr_min)
  return norm_array

def zscore_normalize_array(array):
    """
    Apply zscore normalization to array
    """
    norm_array = stats.zscore(array, axis=None, nan_policy='omit')
    return norm_array

def generate_array(x, gee_dataset_name, save_dir, year, region):
    """
    Generate array from xr dataset
    input x: dataset
    input gee_dataset_name: gee dataset of interest
    """
    var_list = [i for i in x.data_vars]
    multichannel_list = []
    for feature in var_list:
        ds_feature = x[feature]
        arr = ds_feature.to_numpy()
        #norm_arr = zscore_normalize_array(arr)
        multichannel_list.append(arr)

    img = np.array(multichannel_list)
    img = np.nan_to_num(img)

    # Save gee array
    np.save('{}/{}_{}_{}_array.npy'.format(save_dir, gee_dataset_name, year, region), img)


def format_lon(x):
    """
    Format ds longitude to range -180 - 180
    """
    if int(x.coords['lon'].max()) > 180:
        x.coords['lon'] = (x.coords['lon'] + 180) % 360 - 180
    return x.sortby(x.lon)


def filter_bounds(xr_ds, extent):
    """
    Filter xarray bounds
    input: xr_ds: xarray dataset to filter
    imput: extent: extent to clip bounds by
    """
    if extent == 'na':
        min_lat = 20.748
        max_lat = 54.392
        min_lon = -124.875
        max_lon = -70.875
    elif extent == 'eu':
        min_lat = 35.327
        max_lat = 64.485
        min_lon = -9.0
        max_lon = 24.75
    else:
        return xr_ds

    cropped_ds = xr_ds.sel(lat=slice(min_lat, max_lat), lon=slice(min_lon, max_lon))

    return cropped_ds

if __name__ == '__main__':
    args = get_args()
    region = args.region
    gee_dir = args.gee_dir

    if region == 'na':
        full_name = 'NorthAmerica'
    if region == 'eu':
        full_name = 'Europe'
    if region == 'globe':
        full_name = 'Globe'

    # --- Processing Population ---
    years = ['2005','2010','2015', '2020']
    for y in years:
        print('--- Processing Population GEE to array for year {} ---'.format(y))
        ds = xr.open_dataset('{}/pop_population_density_{}_globe_buffersize_55500_with_time.nc'.format(gee_dir, y), engine='netcdf4')

        # Format lon
        ds = format_lon(ds)

        # Subsample over north america
        filt_ds = filter_bounds(ds, region)

        # Flip to the correct orientation
        flipped_ds = filt_ds.reindex(lat=filt_ds.lat[::-1])

        # Make array
        generate_array(flipped_ds, 'population', '/Volumes/PRO-G40/sudsaq/GEE', y, full_name)


    # --- Processing modis ---
    years = ['2005', '2006', '2007', '2008', '2009','2010',
             '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020']
    for y in years:
        print('--- Processing MODIS GEE to array for year {} ---'.format(y))
        ds = xr.open_dataset('{}/modis_LC_Type1_{}_globe_buffersize_55500_with_time.nc'.format(gee_dir, y))
        # Format lon
        ds = format_lon(ds)

        # Subsample over north america
        filtered_ds = filter_bounds(ds, region)

        # Flip to the correct orientation
        flipped_ds = filtered_ds.reindex(lat=filtered_ds.lat[::-1])

        # Make array
        generate_array(flipped_ds, 'modis', '/Volumes/PRO-G40/sudsaq/GEE', y, full_name)







