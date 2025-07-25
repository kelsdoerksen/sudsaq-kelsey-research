"""
Plot for sensitivity mapping
To clean up
"""

import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import argparse
import os

def get_args():
    parser = argparse.ArgumentParser(description='Generate Plots for sensitivity analysis no masking nans')
    parser.add_argument('--unet_dir', help='Directory of UNet run to process fata for')
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

def spatial_map(uq_method, data, region, savedir):
    """
    Spatial map of RF/DL results
    """

    if uq_method in ['DL UQ', 'DL UQ_avg']:
        vmin = 25
        vmax = 50
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
    plt.title('UQ Method: {}'.format(uq_method))
    plt.savefig('{}/{}_spatial_plot.png'.format(savedir, uq_method))
    plt.close()


if __name__ == '__main__':
    args = get_args()
    dl_dir = args.unet_dir

    fns = [x for x in os.listdir(dl_dir) if 'uq_nomask' in x]
    arr_list = []
    for i in range(len(fns)):
        arr_list.append(np.load('{}/{}'.format(dl_dir, fns[i])))

    arr_mean = np.mean(arr_list, axis=0)
    spatial_map('DL UQ', arr_mean, 'na', dl_dir)
