"""
Script for analsysis of RF quantiles
 to compare to DL UQ methods
"""

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from datetime import date, timedelta
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Generate Plots to compare RF quantiles and CQR from UNet')
    parser.add_argument('--unet_dir', help='Directory of UNet run to process fata for')
    parser.add_argument('--test_year', help='Year of test analysis')
    parser.add_argument('--model', help='Model type')
    parser.add_argument('--geo', help='Geographic location. One of na or eu')
    return parser.parse_args()

# --- Defining lon, lat values of regions for plotting
na_lon_vals = [-124.875, -123.75 , -122.625, -121.5  , -120.375, -119.25 ,
           -118.125, -117.   , -115.875, -114.75 , -113.625, -112.5  ,
           -111.375, -110.25 , -109.125, -108.   , -106.875, -105.75 ,
           -104.625, -103.5  , -102.375, -101.25 , -100.125,  -99.   ,
            -97.875,  -96.75 ,  -95.625,  -94.5  ,  -93.375,  -92.25 ,
            -91.125,  -90.   ,  -88.875,  -87.75 ,  -86.625,  -85.5  ,
            -84.375,  -83.25 ,  -82.125,  -81.   ,  -79.875,  -78.75 ,
            -77.625,  -76.5  ,  -75.375,  -74.25 ,  -73.125,  -72.   ,
            -70.875]

na_lat_vals = [20.748, 21.869, 22.991, 24.112, 25.234, 26.355, 27.476, 28.598,
           29.719, 30.841, 31.962, 33.084, 34.205, 35.327, 36.448, 37.57 ,
           38.691, 39.813, 40.934, 42.056, 43.177, 44.299, 45.42 , 46.542,
           47.663, 48.785, 49.906, 51.028, 52.149, 53.271, 54.392]

eu_lat_vals = [35.327, 36.448, 37.57, 38.691, 39.813, 40.934, 42.056, 43.177, 44.299, 45.42, 46.542,
               47.663, 48.785, 49.906, 51.028, 52.149, 53.271, 54.392, 55.514, 56.635, 57.757, 58.878,
               60., 61.121, 62.242, 63.364, 64.485]
eu_lon_vals = [-9., -7.875, -6.75, -5.625, -4.5, -3.375, -2.25, -1.125, 0., 1.125, 2.25, 3.375, 4.5, 5.625, 6.75,
               7.875, 9., 10.125, 11.25, 12.375, 13.5, 14.625, 15.75, 16.875, 18., 19.125, 20.25, 21.375, 22.5,
               23.625, 24.75]

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

def daterange(date1, date2):
    date_list = []
    for n in range(int((date2 - date1).days) + 1):
        dt = date1 + timedelta(n)
        date_list.append(dt.strftime("%Y-%m-%d"))
    return date_list

def preprocess_rf_quantiles(ds, doy, region):
    """
    Preprocess RF quantile data to compare
    """
    ds_095 = ds.sel(time=doy)['0.95']
    ds_005 = ds.sel(time=doy)['0.05']
    length = ds_095 - ds_005

    return length

def get_cqr_length(lower_bound, upper_bound):
    """
    Get length of predictive interval from the upper and
    lower bound predictions
    """
    return np.abs(upper_bound-lower_bound)

def preprocess_cqr_data(data_dir, lower_fns, upper_fns):
    """
    Preprocess CQR data for further plotting
    """
    lower_list = []
    upper_list = []
    for i in range(len(lower_fns)):
        lower_arr = np.load('{}/{}'.format(data_dir,lower_fns[i]))
        upper_arr = np.load('{}/{}'.format(data_dir, upper_fns[i]))

        for j in range(len(lower_arr)):
            lower_list.append(lower_arr[j][0,:,:])
            upper_list.append(upper_arr[j][2,:,:])

    length_list = []
    for k in range(len(lower_list)):
        length = get_cqr_length(lower_list[k], upper_list[k])
        length_list.append(length)
    return length_list


def preprocess_mcd_data(data_dir, fns):
    """
    Preprocess MCD data for further plotting
    """
    pred_list = []
    for i in range(len(fns)):
        arr = np.load('{}/{}'.format(data_dir,fns[i]))

        for j in range(len(arr)):
            pred_list.append(arr[j][0,:,:])

    return pred_list

def spatial_map(uq_method, data, region, doy, savedir):
    """
    Spatial map of RF/DL results
    """

    if uq_method in ['DL UQ', 'DL UQ_avg', 'RF_length']:
        vmin = 20
        vmax = 65
    else:
        vmin = 0
        vmax = 50

    if region == 'na':
        # Hard coded vals
        lon_vals = na_lon_vals
        lat_vals = na_lat_vals

    if region == 'eu':
        lat_vals = eu_lat_vals
        lon_vals = eu_lon_vals

    x, y = np.meshgrid(lon_vals, lat_vals, indexing='xy')
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    plt.pcolor(x, y, data, cmap='coolwarm', vmin=vmin, vmax=vmax)
    ax.coastlines(linewidth=2)
    ax.set_facecolor('gray')
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=1, color='gray', alpha=0.8, linestyle='--')
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    cb = plt.colorbar(pad=0.1)
    plt.tight_layout()
    plt.title('UQ Method: {} for DOY: {}'.format(uq_method, doy))
    plt.savefig('{}/{}_{}_spatial_plot.png'.format(savedir, uq_method, doy))
    plt.close()

if __name__ == '__main__':
    args = get_args()
    dl_dir = args.unet_dir
    year = args.test_year
    model = args.model
    geo = args.geo

    #rf_ds = xr.open_dataset('{}/june.test.quantiles.nc'.format(dl_dir))
    date_list = daterange(date(int(year), 6, 1), date(int(year), 6, 30))

    if model == 'cqr':
        fns_lower = [x for x in os.listdir(dl_dir) if '_lower_cal' in x]
        fns_upper = [x for x in os.listdir(dl_dir) if '_upper_cal' in x]
        fns_lower.sort()
        fns_upper.sort()
        dl_data = preprocess_cqr_data(dl_dir, fns_lower, fns_upper)

    if model == 'mcdropout':
        fns = [x for x in os.listdir(dl_dir) if 'epi_unc' in x]
        date_list = daterange(date(int(year), 6, 1), date(int(year), 6, 30))
        dl_data = preprocess_mcd_data(dl_dir, fns)

        dl_mean = np.mean(dl_data, axis=0)
        spatial_map('DL UQ_avg_mcd', dl_mean, geo, 'June 2019', dl_dir)
        np.save('{}/avg_uq_nomask.npy'.format(dl_dir), dl_mean)

    for d in range(len(date_list)):
        #rf_data = preprocess_rf_quantiles(rf_ds, date_list[d], geo)
        spatial_map('DL UQ', dl_data[d], geo, date_list[d], dl_dir)
        np.save('{}/DL_UQ_{}.npy'.format(dl_dir, date_list[d]), dl_data[d])
        #spatial_map('RF_length', rf_data, geo, date_list[d], dl_dir)

