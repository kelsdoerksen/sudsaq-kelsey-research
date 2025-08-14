"""
Preprocessing scripts for Random Forest local runs
"""

import numpy as np
import xarray as xr
import os
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import datetime as dt
from datetime import datetime


parser = argparse.ArgumentParser(description='Preprocessing RF data')
parser.add_argument("--test_year", help="Specify year to use for test, all other years used for training.")
parser.add_argument("--region", help="Boundary region on Earth to take data. Must be one of: "
                                     "globe, europe, asia, australia, north_america, west_europe, "
                                     "east_europe, west_na, east_na.")
parser.add_argument("--root_dir", help="Root directory for RF data")
parser.add_argument("--momo_data_dir", help="Directory of momo data")
parser.add_argument("--bias_data_dir", help="Directory of bias data")
parser.add_argument("--gee_data_dir", help="Directory of gee data")

args = parser.parse_args()

# Setting directories
print('Setting directories')
root_dir = '/Users/kelseyd/Desktop/random_forest/data'
momo_data_dir = '/Volumes/MLIA_active_data/data_SUDSAQ/data/momo'
gee_data_dir = '/Volumes/MLIA_active_data/data_SUDSAQ/gee'
bias_data_dir = '/Volumes/MLIA_active_data/data_SUDSAQ/summaries/bias/gattaca.v4.bias-median.extended/combined_data'

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

# Top features from July RF experiments
feature_list = ['momo.2dsfc.NH3', 'momo.2dsfc.PROD.HOX', 'momo.2dsfc.DMS', 'momo.co',
                'momo.2dsfc.HNO3', 'momo.2dsfc.BrONO2', 'momo.t',
                'momo.no2', 'momo.2dsfc.PAN', 'momo.ps', 'momo.2dsfc.HO2',
                'momo.2dsfc.C5H8', 'momo.olrc', 'momo.oh', 'momo.slrc', 'momo.so2']

# --- Hard-coding train years to be 2005-2016, test year 2017 ---
train_years = [2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015]
test_year = [2016]


def format_lon(x):
    '''
    Format longitude to use for subdividing regions
    to be -180 to 180
    Input: xarray dataset
    Output: xarray dataset
    '''
    x.coords['lon'] = (x.coords['lon'] + 180) % 360 - 180
    return x.sortby(x.lon)


def daily(ds):
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

def daily_mda8(label_ds):
    """
    Filter for daily mda8 estimate (measurement once every 24 hours)
    label_ds: xarray to generate labels from
    m: month of query
    """
    # Select timestamp starting with startime and increment by one day
    start_date = list(label_ds.coords['time'].values)[0]
    print('Generating labels for year {}'.format(str(start_date)[0:4]))
    increment = 24
    label_list = []
    m_length = int(len(label_ds)/12)

    for d in range(0, m_length):
        t = start_date + np.timedelta64(increment*d, 'h')
        sub_ds = label_ds.sel(time=t)
        label_list.append(sub_ds)

    return label_list


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


def generate_features(x):
    """
    x: xarray dataset
    """
    features = x[feature_list]
    daily_ds = daily(features)
    data = daily_ds.to_array().stack({'loc': ['lat', 'lon', 'time']})
    data = data.transpose('loc', 'variable')
    df = pd.DataFrame(data=data.values, columns=feature_list)
    return df


def generate_labels(x):
    """
    x: xarray dataset
    """
    target = x['momo.mda8']
    label_list = daily_mda8(target)
    daily_labels = []
    for label in label_list:
        label = label.to_dataset()
        data = label.to_array().stack({'loc': ['lat', 'lon']})
        data = data.transpose('loc', 'variable')
        label_array = data.values
        daily_labels.append(label_array[:,0])
    final_labels = np.concatenate(daily_labels)

    return final_labels


def generate_ozone_training_data(t_years, geographic_extent, n_features):
    """
    Load training samples for training years, aoi
    """

    months_dict = {'06': 'june',
                   '07': 'july',
                   '08': 'aug'}

    if geographic_extent == 'eu':
        full_geo_name = 'Europe'
    if geographic_extent == 'na':
        full_geo_name = 'NorthAmerica'

    for m in months_dict.keys():
        monthly_label = []
        print('Generating training samples for month: {}'.format(m))
        monthly_features = []
        for year in t_years:
            print('Processing year {}'.format(year))
            ds = xr.open_mfdataset(['{}/{}/{}.nc'.format(momo_data_dir,year, m)])
            # Format lon
            ds = format_lon(ds)

            # Filter area based on geographic extent
            ds = filter_bounds(ds, geographic_extent)

            # Generate features
            features = generate_features(ds)
            monthly_features.append(features)

            # Generate labels

            #labels = generate_labels(ds)
            #monthly_label.append(labels)

        # Save features

        df_features = pd.concat(monthly_features)
        df_features.to_csv('{}/samples/{}/{}features/{}_{}-{}_features.csv'.format(root_dir, full_geo_name, n_features,
                                                                                       months_dict[m],
                                                                 t_years[0],t_years[-1]))
        '''

        # Save labels
        target_df = pd.DataFrame(np.concatenate(monthly_label), columns=['target'])
        target_df.to_csv('{}/target/{}/mda8/{}_{}-{}_target.csv'.format(root_dir, full_geo_name, months_dict[m],
                                                             train_years[0],train_years[-1]))
        '''

def generate_ozone_testing_data(testing_year, geographic_extent, n_features):
    """
    Similar to train script but for one year
    Currently hardcoded to 2016 to match UNet model
    """
    months_dict = {'06': 'june',
                   '07': 'july',
                   '08': 'aug'}

    if geographic_extent == 'eu':
        full_geo_name = 'Europe'
    if geographic_extent == 'na':
        full_geo_name = 'NorthAmerica'

    for m in months_dict.keys():
        monthly_label = []
        monthly_features = []
        ds = xr.open_mfdataset(['{}/{}/{}.nc'.format(momo_data_dir, testing_year, m)])

        # Format lon
        ds = format_lon(ds)

        # Filter area based on geography specified
        ds = filter_bounds(ds, geographic_extent)

        # Generate features
        features = generate_features(ds)
        monthly_features.append(features)

        # Generate labels
        #labels = generate_labels(ds)
        #monthly_label.append(labels)

        # Save features
        df_features = pd.concat(monthly_features)
        df_features.to_csv('{}/samples/{}/{}features/{}_{}_features.csv'.format(root_dir, full_geo_name, n_features,
                                                                     months_dict[m], testing_year))
        '''
        # Save labels
        target_df = pd.DataFrame(np.concatenate(monthly_label), columns=['target'])
        target_df.to_csv('{}/target/{}/mda8/{}_{}_target.csv'.format(root_dir, full_geo_name,
                                                                     months_dict[m], testing_year))
        '''

def daily_toar(x, month_length):
    """
    Get list of daily toar for calcs with mda8
    later on
    """
    toar_list = []
    for i in range(month_length):
        toar_sub = x.sel(time=x.coords['time'].values[i])
        toar_list.append(toar_sub)

    return toar_list


def generate_bias_labels(year, month, geo_extent):
    """
    Generate bias label for year, month of query for specified extent
    bias = momo.mda8 - toar.o3.dma8epa.median
    """

    if month == 'june':
        month_length = 30
        month_int = '06'
    if month == 'july':
        month_length = 31
        month_int = '07'
    if month == 'aug':
        month_length = 31
        month_int = '08'

    # Load momo ds
    momo_ds = xr.open_mfdataset(["/Volumes/MLIA_active_data/data_SUDSAQ/data/momo/{}/{}.nc".format(year, month_int)])

    # Format lon
    momo_ds = format_lon(momo_ds)

    # Subset for geographic extent, either NA or EU supported
    filtered_ds = filter_bounds(momo_ds, geo_extent)

    # Get daily mda8
    target = filtered_ds['momo.mda8']
    mda8_list = daily_mda8(target)

    # Load toar
    toar = xr.open_mfdataset(["/Volumes/MLIA_active_data/data_SUDSAQ/data/toar/matched/lon360/{}/{}.nc".
                             format(year, month_int)])
    # Format lon
    toar = format_lon(toar)
    # Filter to bounds
    toar_filtered = filter_bounds(toar, geo_extent)
    # Subselect median calc
    toar_med = toar_filtered['toar.o3.dma8epa.median']

    toar_med_list = daily_toar(toar_med, month_length)

    # Calculate bias
    daily_labels = []
    for i in range(len(toar_med_list)):
        bias = mda8_list[i] - toar_med_list[i]
        bias_ds = bias.to_dataset(name='bias')
        data = bias_ds.to_array().stack({'loc': ['lat', 'lon']})
        data = data.transpose('loc', 'variable')
        label_array = data.values
        daily_labels.append(label_array[:, 0])
    final_labels = np.concatenate(daily_labels)

    return final_labels


def generate_bias_data(t_years, geographic_extent):
    """
    Generate bias data for RF target
    """
    months_dict = {'06': 'june'}
    t_years = [2005, 2006]

    if geographic_extent == 'eu':
        full_geo_name = 'Europe'
    if geographic_extent == 'na':
        full_geo_name = 'NorthAmerica'

    for m in months_dict.keys():
        print('Processing for month: {}'.format(months_dict[m]))
        monthly_label = []
        for year in t_years:
            print('Processing year {}'.format(year))
            labels = generate_bias_labels(year, months_dict[m], geographic_extent)
            monthly_label.append(labels)

        target_df = pd.DataFrame(np.concatenate(monthly_label), columns=['target'])
        if len(t_years) > 1:
            target_df.to_csv('{}/target/{}/bias/{}_{}-{}_target.csv'.format(root_dir, full_geo_name, months_dict[m],
                                                                            t_years[0], t_years[-1]))
        else:
            target_df.to_csv('{}/target/{}/bias/{}_{}_target.csv'.format(root_dir, full_geo_name, months_dict[m],
                                                                            t_years[0]))



geo = 'eu'

# --- Running Scripts ---

# mda8
'''
num_features = 16
# Generate momo training, testing data
generate_ozone_training_data(train_years, geo, num_features)
generate_ozone_testing_data(test_year, geo, num_features)
'''

# Generate bias labels
generate_bias_data(train_years, geo)
generate_bias_data(test_year, geo)

