"""
Script for processing the momochem data into n-channel
images and labels over North American extent for UNet
"""

import numpy as np
import xarray as xr
import datetime as dt
import argparse
from scipy import stats
import math


def get_args():
    parser = argparse.ArgumentParser(description='Preprocessing MOMOChem for UNet')
    parser.add_argument('--month', help='Month of query')
    parser.add_argument('--region', help='Location, na, eu or global')
    parser.add_argument('--year', help='Year of query, supports 2005-2020')
    parser.add_argument('--sample_dir', help='Directory that sample data is stored to query')
    parser.add_argument('--label_dir', help='Directory that label data is stored to query')
    parser.add_argument('--save_dir', help='Save directory of data')
    parser.add_argument('--remove_feature', help='Feature to remove for sensitivity analysis')
    return parser.parse_args()

month_dict = {
    '06': 'june',
    '07': 'july',
    '08': 'august'
}

'''
# Old: Top nine features from July RF experiments
feature_list = ['momo.2dsfc.NH3', 'momo.2dsfc.PROD.HOX', 'momo.2dsfc.DMS', 'momo.co',
                'momo.2dsfc.HNO3', 'momo.2dsfc.BrONO2', 'momo.t', 'momo.no2', 'momo.2dsfc.PAN']
# Added features Science PIs identified
feature_list = ['momo.ps', 'momo.2dsfc.HO2', 'momo.2dsfc.C5H8', 'momo.olrc', 'momo.oh', 'momo.slrc', 'momo.so2']
'''

'''
Timezones = [
# offset, ( west, east)
    (  0, (  0.0, 7.5)),
    (  1, (  7.5, 22.5)),
    (  2, ( 22.5, 37.5)),
    (  3, ( 37.5, 52.5)),
    (  4, ( 52.5, 67.5)),
    (  5, ( 67.5, 82.5)),
    (  6, ( 82.5, 97.5)),
    (  7, ( 97.5, 112.5)),
    (  8, (112.5, 127.5)),
    (  9, (127.5, 142.5)),
    ( 10, (142.5, 157.5)),
    ( 11, (157.5, 172.5)),
    ( 12, (172.5, 180.0)),
    (-12, (180.0, 187.5)),
    (-11, (187.5, 202.5)),
    (-10, (202.5, 217.5)),
    ( -9, (217.5, 232.5)),
    ( -8, (232.5, 247.5)),
    ( -7, (247.5, 262.5)),
    ( -6, (262.5, 277.5)),
    ( -5, (277.5, 292.5)),
    ( -4, (292.5, 307.5)),
    ( -3, (307.5, 322.5)),
    ( -2, (322.5, 337.5)),
    ( -1, (337.5, 352.5)),
    (  0, (352.5, 360.0))
]
'''

Timezones = [
    (0, (0.0, 7.5)),
    (1, (7.5, 22.5)),
    (2, (22.5, 37.5)),
    (3, (37.5, 52.5)),
    (4, (52.5, 67.5)),
    (5, (67.5, 82.5)),
    (6, (82.5, 97.5)),
    (7, (97.5, 112.5)),
    (8, (112.5, 127.5)),
    (9, (127.5, 142.5)),
    (10, (142.5, 157.5)),
    (11, (157.5, 172.5)),
    (12, (172.5, 180.0)),
    (-12, (-180.0, -172.5)),
    (-11, (-172.5, -157.5)),
    (-10, (-157.5, -142.5)),
    (-9, (-142.5, -127.5)),
    (-8, (-127.5, -112.5)),
    (-7, (-112.5, -97.5)),
    (-6, (-97.5, -82.5)),
    (-5, (-82.5, -67.5)),
    (-4, (-67.5, -52.5)),
    (-3, (-52.5, -37.5)),
    (-2, (-37.5, -22.5)),
    (-1, (-22.5, -7.5)),
    (0, (-7.5, 0.0))
]

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
    norm_array = stats.zscore(array, axis=None, nan_policy = 'omit')
    return norm_array

def generate_samples(sample_ds, feature_list, save_dir):
    """
    Generate n-channel array samples from nc momochem xarrays.
    Pseudo code:
    1. filter dataset for only top features from feature_list
    2. For each day, make an array per variable -> can perhaps store
    this in a list for each variable, so would have 9 lists, one per var
    of length 31 for each day in July
    """
    date_list = sample_ds.coords['time'].values.tolist()
    date_list = [np.datetime64(x, "ns") for x in date_list]

    # iterate through each day in dataset per feature
    for doy in date_list:
        print('Processing date: {}'.format(doy))
        multi_channel_list = []
        for feature in feature_list:
            print('Processing feature: {}'.format(feature))
            ds_filt = sample_ds.sel(indexers={'time': doy})
            ds_feature = ds_filt[feature]
            if math.isnan(np.amax(ds_feature.to_numpy())):
                ds_feature = ds_feature.interpolate_na(dim='lon')
                ds_feature = ds_feature.interpolate_na(dim='lat')
                print('Linear interpolating NaNs for feature: {}'.format(feature))
            ds_feature = ds_feature.reindex(lat=ds_feature.lat[::-1])   # Flipping so its oriented correctly
            arr = ds_feature.to_numpy()
            #norm_arr = zscore_normalize_array(arr)
            multi_channel_list.append(arr)
        save_name_date = str(doy)[0:10]
        # stack all arrays in multi_channel_list to make multi-channel image
        img = np.array(multi_channel_list)
        np.save('{}/{}_sample'.format(save_dir, save_name_date), img)


def generate_labels(label_ds, save_dir):
    """
    Generate array labels from nc momochem xarrays
    """
    date_list = label_ds.coords['time'].values.tolist()
    date_list = [np.datetime64(x, "ns") for x in date_list]

    # iterate through each day in dataset per feature
    for doy in date_list:
        save_name_date = str(doy)[0:10]
        ds_filt = label_ds.sel(indexers={'time': doy})
        ds_filt = ds_filt.reindex(lat=ds_filt.lat[::-1])    # Flipping so its oriented correctly
        label_np = ds_filt.to_array()
        print('Saving file: {}'.format(save_name_date))
        np.save('{}/{}_label'.format(save_dir, save_name_date), label_np)


def daily(ds, feature_list):
    """
    Aligns a dataset to a daily average
    """
    def select_times(ds, sel, time):
        """
        Selects timestamps using integer hours (0-23) over all dates
        """
        if isinstance(sel, list):
            mask = (dt.time(sel[0]) <= time) & (time <= dt.time(sel[1]))
            ds   = ds.where(mask, drop=True)
            # Floor the times to the day for the groupby operation
            ds.coords['time'] = ds.time.dt.floor('1D')

            # Now group as daily taking the mean
            ds = ds.groupby('time').mean()
        else:
            mask = (time == dt.time(sel))
            ds   = ds.where(mask, drop=True)

            # Floor the times to the day for the groupby operation
            ds.coords['time'] = ds.time.dt.floor('1D')

        return ds

    # Select time ranges per config
    time = ds.time.dt.time
    data = []
    time_range = [8, 15]
    print('-- Using local timezones')
    ns = ds[feature_list]
    local = []
    for offset, bounds in Timezones:
        print('Running bounds: {}'.format(bounds))
        sub  = ns.sel(lon=slice(*bounds))
        time = ( sub.time + np.timedelta64(offset, 'h') ).dt.time
        sub  = select_times(sub, time_range, time)
        local.append(sub)

    # Merge these datasets back together to create the full grid
    print('-- Merging local averages together (this can take awhile)')
    data.append(xr.merge(local))
    # Add variables that don't have a time dimension back in
    timeless = ds.drop_dims('time')
    if len(timeless) > 0:
        print(f'- Appending {len(timeless)} timeless variables: {list(timeless)}')
        data.append(timeless)

    # Merge the selections together
    print('- Merging all averages together')
    ds = xr.merge(data)

    # Cast back to custom Dataset (xr.merge creates new)
    ds = xr.Dataset(ds)

    return ds


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
    if extent == 'globe':
        return xr_ds

    cropped_ds = xr_ds.sel(lat=slice(min_lat, max_lat), lon=slice(min_lon, max_lon))

    return cropped_ds


def daily_mda8(label_ds, m_length):
    """
    Filter for daily mda8 estimate (measurement once every 24 hours)
    label_ds: xarray to generate labels from
    m: month of query
    """
    # Select timestamp starting with startime and increment by one day
    start_date = label_ds.coords['time'].values.tolist()[0]
    print('Generating labels for year {}'.format(str(start_date)[0:4]))
    increment = 24
    label_list = []

    for d in range(0, m_length):
        t = start_date + np.timedelta64(increment*d, 'h')
        sub_ds = label_ds.sel(time=t)
        label_list.append(sub_ds['momo.mda8'])

    return label_list


def format_lon(x):
    """
    Format ds longitude to range -180 - 180
    """
    if int(x.coords['lon'].max()) > 180:
        x.coords['lon'] = (x.coords['lon'] + 180) % 360 - 180
    return x.sortby(x.lon)


def run_generate_samples(x, m, features, save_dir):
    """
    :param: x: xarray of data
    param: m: month of query
    param: featrues: list of features from data
    """
    print('--- Generating n-channel samples ---')
    # Daily data
    #daily_ds = daily(x, features)
    # Load ds
    #daily_ds.load()
    # Generate the n-channel sample
    generate_samples(x, features, save_dir)


def run_generate_labels(x, m, save_dir):
    """
    x: xarray
    m: month of query
    extent: geographic extent of label
   """
    print('--- Generating labels ---')
    month_length = None
    if m == 'june':
        month_length = 30
    if m == 'july':
        month_length = 31
    if m == 'aug':
        month_length = 31

    #labels = daily_mda8(x, month_length)
    generate_labels(x, save_dir)

if __name__ == '__main__':
    args = get_args()
    sample_dir = args.sample_dir
    label_dir = args.label_dir
    month = args.month
    year = args.year
    geo_extent = args.region
    save_dir = args.save_dir

    print('--- Loading Data for year {}---'.format(year))

    # ----- Generate samples

    sample_ds = xr.open_dataset('{}/{}/{}/test.data.nc'.format(sample_dir, month, year))
    # Get list of features from xarray
    features = [i for i in sample_ds.data_vars]
    # Format lon
    sample_ds = format_lon(sample_ds)
    # Subsample over extent
    filt_sample_ds = filter_bounds(sample_ds, geo_extent)
    run_generate_samples(filt_sample_ds, month, features, save_dir)

    # ----- Generate labels
    label_ds = xr.open_dataset('{}/{}/{}/test.target.nc'.format(label_dir, month, year))
    label_ds = format_lon(label_ds)
    filt_label_ds = filter_bounds(label_ds, geo_extent)
    run_generate_labels(filt_label_ds, month, save_dir)