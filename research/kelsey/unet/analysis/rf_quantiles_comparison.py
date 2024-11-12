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

globe_lat_vals = [-89.142, -88.029, -86.911, -85.791, -84.67, -83.549, -82.428,
                  -81.307, -80.185, -79.064, -77.943, -76.821, -75.7, -74.578,
                  -73.457, -72.336, -71.214, -70.093, -68.971, -67.85, -66.728,
                  -65.607, -64.485, -63.364, -62.242, -61.121, -60.0, -58.878,
                  -57.757, -56.635, -55.514, -54.392, -53.271, -52.149, -51.028,
                  -49.906, -48.785, -47.663, -46.542, -45.42, -44.299, -43.177,
                  -42.056, -40.934, -39.813, -38.691, -37.57, -36.448, -35.327,
                  -34.205, -33.084, -31.962, -30.841, -29.719, -28.598, -27.476,
                  -26.355, -25.234, -24.112, -22.991, -21.869, -20.748, -19.626,
                  -18.505, -17.383, -16.262, -15.14, -14.019, -12.897, -11.776,
                  -11.654,  -9.5327, -8.4112,  -7.2897,  -6.1682,  -5.0467,
                  -3.9252,  -2.8037, -1.6822, -0.56074, 0.56074, 1.6822, 2.8037,
                  3.9252, 5.0467, 6.1682, 7.2897, 8.4112, 9.5327, 10.654, 11.776,
                  12.897,  14.019,  15.14,  16.262,  17.383, 18.505, 19.626, 20.748,
                  21.869,  22.991,  24.112, 25.234, 26.355, 27.476, 28.598, 29.719,
                  30.841, 31.962, 33.084, 34.205, 35.327, 36.448, 37.57, 38.691,
                  39.813, 40.934, 42.056, 43.177, 44.299, 45.42, 46.542, 47.663,
                  48.785,  49.906,  51.028, 52.149, 53.271, 54.392, 55.514, 56.635,
                  57.757, 58.878, 60.0, 61.121, 62.242, 63.364, 64.485, 65.607,
                  66.728,  67.85,  68.971,  70.093,  71.214, 72.336, 73.457, 74.578,
                  75.7,  76.821,  77.943, 79.064, 80.183, 81.307, 82.428, 83.549,
                  84.67, 85.791, 86.911, 88.029, 89.142]
globe_lon_vals = [-180.0, -178.875, -177.75, -176.625, -175.5, -174.375, -173.25,
                  -172.125, -171.0, -169.875, -168.75, -167.625, -166.5, -165.375,
                  -164.25, -163.125, -162.0, -160.875, -159.75, -158.625, -157.5,
                  -156.375, -155.25, -154.125, -153.0, -151.875, -150.75, -149.625,
                  -148.5, -147.375, -146.25, -145.125, -144.0, -142.875, -141.75,
                  -140.625, -139.5, -138.375, -137.25, -136.125, -135.0, -133.875,
                  -132.75, -131.625, -130.5, -129.375, -128.25, -127.125, -126.0,
                  -124.875, -123.75, -122.625, -121.5, -120.375, -119.25, -118.125,
                  -117.0, -115.875, -114.75, -113.625, -112.5, -111.375, -110.25,
                  -109.125, -108.0, -106.875, -105.75, -104.625, -103.5, -102.375,
                  -101.25, -100.125, -99.0, -97.875, -96.75, -95.625,  -94.5, -93.375,
                  -92.25, -91.125, -90.0, -88.875,  -87.75, -86.625, -85.5, -84.375,
                  -83.25, -82.125,  -81.0, -79.875, -78.75, -77.625, -76.5, -75.375,
                  -74.25, -73.125, -72.0, -70.875, -69.75, -68.625,  -67.5, -66.375,
                  -65.25, -64.125, -63.0, -61.875,  -60.75, -59.625, -58.5, -57.375,
                  -56.25, -55.125,  -54.0, -52.875, -51.75, -50.625, -49.5, -48.375,
                  -47.25, -46.125, -45.0, -43.875, -42.75, -41.625,  -40.5, -39.375,
                  -38.25, -37.125, -36.0, -34.875,  -33.75, -32.625, -31.5, -30.375,
                  -29.25, -28.125,  -27.0, -25.875, -24.75, -23.625, -22.5, -21.375,
                  -20.25, -19.125, -18.0, -16.875, -15.75, -14.625, -13.5, -12.375,
                  -11.25, -10.125, -9.0, -7.875, -6.75, -5.625, -4.5, -3.375, -2.25,
                  -1.125, 0.0, 1.125, 2.25, 3.375, 4.5, 5.625, 6.75 , 7.875, 9.0,
                  10.125, 11.25, 12.375, 13.5, 14.625, 15.75, 16.875, 18.0, 19.125,
                  20.25, 21.375, 22.5, 23.625, 24.75, 25.875, 27.0, 28.125, 29.25,
                  30.375, 31.5, 32.625, 33.75, 34.875, 36.0, 37.125,   38.25, 39.375,
                  40.5, 41.625, 42.75, 43.875, 45.0, 46.125, 47.25, 48.375, 49.5,
                  50.625, 51.75, 52.875, 54.0, 55.125, 56.25, 57.375, 58.5, 59.625,
                  60.75, 61.875, 63.0, 64.125, 65.25, 66.375, 67.5, 68.625, 69.75,
                  70.875, 72.0, 73.125, 74.25, 75.375, 76.5, 77.625, 78.75, 79.875,
                  81.0, 82.125, 83.25, 84.375, 85.5, 86.625, 87.75, 88.875, 90.0,
                  91.125, 92.25, 93.375, 94.5, 95.625, 96.75, 97.875, 99.0, 100.125,
                  101.25, 102.375, 103.5, 104.625, 105.75, 106.875, 108.0, 109.125,
                  110.25, 111.375, 112.5, 113.625, 114.75, 115.875, 117.0, 118.125,
                  119.25, 120.375, 121.5,122.625, 123.75, 124.875, 126.0, 127.125,
                  128.25, 129.375, 130.5, 131.625, 132.75, 133.875, 135.0, 136.125,
                  137.25, 138.375, 139.5, 140.625, 141.75, 142.875, 144.0, 145.125,
                  146.25, 147.375,  148.5, 149.625, 150.75, 151.875, 153.0, 154.125,
                  155.25, 156.375, 157.5, 158.625, 159.75, 160.875, 162.0, 163.125,
                  164.25, 165.375, 166.5, 167.625,  168.75, 169.875, 171.0, 172.125,
                  173.25, 174.375, 175.5, 176.625, 177.75, 178.875]

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
    ds_doy = ds.sel(time=doy)['0.9']
    ds_region = filter_bounds(ds_doy, region)
    return ds_region

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

def spatial_map(uq_method, data, region, doy, savedir):
    """
    Spatial map of RF/DL results
    """

    if uq_method == 'DL UQ':
        vmin = 10
        vmax = 70
    else:
        vmin = 10
        vmax = 70

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

    # Hardcoding for June 2016
    rf_ds = xr.open_dataset('/Volumes/PRO-G40/sudsaq/unet/data/RF_Quantiles/NorthAmerica_Model/june.test.quantiles.nc')
    fns_lower = [x for x in os.listdir(dl_dir) if '_lower_cal' in x]
    fns_upper = [x for x in os.listdir(dl_dir) if '_upper_cal' in x]
    fns_lower.sort()
    fns_upper.sort()

    date_list = daterange(date(int(year),6,1), date(int(year),6,30))
    dl_data = preprocess_cqr_data(dl_dir, fns_lower, fns_upper)

    for d in range(len(date_list)):
        rf_data = preprocess_rf_quantiles(rf_ds, date_list[d], 'na')
        spatial_map('DL UQ', dl_data[d], 'na', date_list[d], dl_dir)
        spatial_map('RF_0.9_Quantile', rf_data, 'na', date_list[d], dl_dir)

