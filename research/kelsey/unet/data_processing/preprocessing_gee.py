"""
Script for processing the GEE features to be array format
to use in n-channel image input for UNet
"""

import numpy as np
import xarray as xr
from numpy import moveaxis
import datetime as dt
import os
from scipy import stats

"""
Pseudo-code:
1. Read in momo data
2. Subset over region of interest
3. Save

-> I think it makes more sense to just append the data based on the date of the other data each time I want to run 
it, (or rather, let's generate the gee stuff first as arrays and then run another script to append but have this
done separately so I can keep the individual datasets
"""

root_dir = "/Volumes/MLIA_active_data/data_SUDSAQ/gee_correct"
root_save_dir = "/Users/kelseyd/Desktop/unet/data/NorthAmerica/zscore_normalization/gee"

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
    norm_array = stats.zscore(array, axis=None)
    return norm_array

def generate_array(x, gee_dataset_name):
    """
    Generate array from xr dataset
    input x: dataset
    input gee_dataset_name: gee dataset of interest
    """
    var_list = [i for i in x.data_vars]
    dummy_date = x.time.values[0]   # Dummy date to filter since GEE modis, pop is constant vals
    subset_ds = x.sel(time=dummy_date)
    multichannel_list = []
    for feature in var_list:
        ds_feature = subset_ds[feature]
        arr = ds_feature.to_numpy()
        norm_arr = zscore_normalize_array(arr)
        multichannel_list.append(norm_arr)

    save_name_date = str(dummy_date)[0:4]
    img = np.array(multichannel_list)

    # Save gee array
    np.save('{}/{}/{}_{}_array'.format(root_save_dir, gee_dataset_name, save_name_date, gee_dataset_name), img)


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
    if extent == 'eu':
        min_lat = 35.327
        max_lat = 64.485
        min_lon = -9.0
        max_lon = 24.75

    cropped_ds = xr_ds.sel(lat=slice(min_lat, max_lat), lon=slice(min_lon, max_lon))

    return cropped_ds


# --- Processing Population ---
years = ['2005','2010','2015']
for y in years:
    print('--- Processing Population GEE to array for year {} ---'.format(y))
    ds = xr.open_mfdataset(["/Volumes/MLIA_active_data/data_SUDSAQ/gee_correct/population/"
                            "pop_{}_globe_buffersize_55500_with_time.nc".format(y)])

    # Format lon
    ds = format_lon(ds)

    # Subsample over north america
    filt_ds = filter_bounds(ds, 'na')

    # Make array
    generate_array(filt_ds, 'population')

# --- Processing modis ---
years = ['2005', '2006', '2007', '2008', '2009','2010',
         '2011', '2012', '2013', '2014', '2015', '2016']
for y in years:
    print('--- Processing MODIS GEE to array for year {} ---'.format(y))
    ds = xr.open_mfdataset(["/Volumes/MLIA_active_data/data_SUDSAQ/gee_correct/modis/"
                            "modis_{}_globe_buffersize_55500_with_time.nc".format(y)])
    # Format lon
    ds = format_lon(ds)

    # Subsample over north america
    filtered_ds = filter_bounds(ds, 'na')

    # Make array
    generate_array(filtered_ds, 'modis')







