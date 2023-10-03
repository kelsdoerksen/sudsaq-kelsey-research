"""
Script for processing bias as labels for UNet
"""

import xarray as xr
import numpy as np

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

def preprocess_summary_bias():
    """
    Functionality to preprocess bias from
    the summaries folder
    """
    save_dir = '/Users/kelseyd/Desktop/unet/data/NorthAmerica/bias/august/2005-2015_labels'

    # Load bias median target
    ds = xr.open_mfdataset(['/Volumes/MLIA_active_data/data_SUDSAQ/summaries/bias/gattaca.v4.bias-median.extended/combined_data/aug/test.target.nc'])
    # Format lon if necessary
    ds_lon = format_lon(ds)
    # Subset to NA
    ds_na = filter_bounds(ds_lon)

    # Make labels per date
    for day in range(len(ds_na['time'])):
        date = ds_na['time'][day]
        label = ds_na.sel(time=date)
        label_darray = label.to_array()
        arr = label_darray.to_numpy()
        save_name_date = str(label.coords['time'].values)[0:10]
        np.save('{}/{}_label'.format(save_dir, save_name_date), arr[0, :, :])

def daily_mda8(label_ds, m_length):
    """
    Filter for daily mda8 estimate (measurement once every 24 hours)
    label_ds: xarray to generate labels from
    m: month of query
    """
    # Select timestamp starting with startime and increment by one day
    start_date = list(label_ds.coords['time'].values)[0]
    #print('Generating labels for year {}'.format(str(start_date)[0:4]))
    increment = 24
    label_list = []

    for d in range(0, m_length):
        t = start_date + np.timedelta64(increment*d, 'h')
        sub_ds = label_ds.sel(time=t)
        label_list.append(sub_ds['momo.mda8'])

    return label_list

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


def preprocess_bias_toar(year, month, geo_extent, results_dir):
    """
    Preprocess bias by calculating difference between mda8 and
    toar measurement
    input: year: year of query
    input: month: month of query
    input: results_dir: where to save labels
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
    mda8_list = daily_mda8(filtered_ds, month_length)

    # Load toar
    toar = xr.open_mfdataset(["/Volumes/MLIA_active_data/data_SUDSAQ/data/toar/matched/lon360/{}/{}.nc".
                             format(year, month_int)])
    # Format lon
    toar = format_lon(toar)
    # Filter to NA bounds
    toar_filtered = filter_bounds(toar, geo_extent)
    # Subselect median calc
    toar_med = toar_filtered['toar.o3.dma8epa.median']

    toar_med_list = daily_toar(toar_med, month_length)

    dates = toar_med['time'].values

    # Calculate bias
    for i in range(len(toar_med_list)):
        bias = mda8_list[i] - toar_med_list[i]
        bias_arr = bias.to_numpy()
        # Save as np array
        save_name_date = str(dates[i])[0:10]
        np.save('{}/{}_label'.format(results_dir, save_name_date), bias_arr)


years = ['2005','2006', '2007', '2008','2009', '2010', '2011', '2012',
         '2013', '2014', '2015', '2016']
months = ['june', 'july', 'aug']
geog = 'eu'

local_dict = {
    'eu': 'Europe',
    'na': 'NorthAmerica'
}

for y in years:
    for m in months:
        save_directory = '/Users/kelseyd/Desktop/unet/data/{}/labels_bias/{}'.format(local_dict[geog], m)
        preprocess_bias_toar(y, m, geog, save_directory)

# Below old, querying from the exsisting bias calcs (no need to do things twice)
#preprocess_summary_bias()
