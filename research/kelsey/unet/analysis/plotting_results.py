"""
Script for plotting different results for nicer viz
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy import stats
from scipy.stats import ks_2samp        # use to quantify the difference of two distributions
import argparse
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from datetime import datetime, timedelta, date
import pickle
from sklearn.metrics import r2_score
import xarray as xr


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

def daterange(date1, date2):
  date_list = []
  for n in range(int ((date2 - date1).days)+1):
    dt = date1 + timedelta(n)
    date_list.append(dt.strftime("%Y-%m-%d"))
  return date_list


def get_args():
    parser = argparse.ArgumentParser(description='Generate Plots from UNet Results')
    parser.add_argument('--save_dir', help='Save Directory')
    parser.add_argument('--target', help='Model target (bias or mda8)')
    parser.add_argument('--channels', help='Number of channels')
    parser.add_argument('--region', help='Region')
    parser.add_argument('--model', help='Model type, standard, mcdropout or cqr')
    parser.add_argument('--analysis_period', help='Time of analysis period, june,'
                                                  'july, aug, summer')
    parser.add_argument('--sensitivity', help='Specify if sensitivty analysis',
                        required=False, default=None)
    return parser.parse_args()


bbox_dict = {'globe':[-180, 180, -90, 90],
            'europe': [-20, 40, 25, 80],
            'asia': [110, 160, 10, 70],
            'australia': [130, 170, -50, -10],
            'north_america': [-140, -50, 10, 80],
            'west_europe': [-20, 10, 25, 80],
            'east_europe': [10, 40, 25, 80],
            'west_na': [-140, -95, 10, 80],
            'east_na': [-95, -50, 10, 80],
            'east_europe1': [20, 35, 40, 50]}

def truth_vs_predicted(target, predict, dir, region):
    """
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    # Retrieve the limits and expand them by 5% so everything fits into a square grid
    limits = min([min(target), min(predict)]), max([max(target), max(predict)])
    limits = limits[0] - np.abs(limits[0] * .05), limits[1] + np.abs(limits[1] * .05)
    ax.set_ylim(limits)
    ax.set_xlim(limits)

    # Create the horizontal line for reference
    ax.plot((limits[0], limits[1]), (limits[0], limits[1]), '--', color='r')

    # Create the density values
    kernel = stats.gaussian_kde([target, predict])
    density = kernel([target, predict])

    plot = ax.scatter(target, predict, c=density, cmap='viridis', s=5)

    # Create the colorbar without ticks
    cbar = fig.colorbar(plot, ax=ax)
    cbar.set_ticks([])

    # Set labels
    cbar.set_label('Density')
    ax.set_xlabel('Truth')
    ax.set_ylabel('Predicted')
    ax.set_title('Truth vs Predicted for {}'.format(region))

    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('{}/truth_vs_pred.png'.format(dir))
    plt.close()
    #plt.show()

def plot_histogram(groundtruth, pred, dir, region, target_var):
    '''
    Plot histogram of true vs predicted
    '''
    bins = np.linspace(-120, 120, 300)
    plt.hist(groundtruth, bins, histtype='step', label=['target'])
    plt.hist(pred, bins, histtype='step', label=['prediction'])
    plt.legend(loc='upper right')
    plt.xlabel('{}'.format(target_var))
    plt.ylabel('count')

    # save histogram so I can plot it with rf later
    df = pd.DataFrame()
    df['pred'] = pred
    df['gt'] = groundtruth
    df.to_csv('{}/histogram_data.csv'.format(dir))

    # to update not to be hardcoded
    plt.ylim(0, 2000)

    plt.title('Truth vs Predicted Histogram for {}'.format(region))

    # Calculate Kolmogorov-Smirnov test, assuming continuous distribution
    ks_val = ks_2samp(pred, groundtruth)

    plt.savefig('{}/truth_vs_pred_hist.png'.format(dir))
    #plt.show()
    plt.close()

def get_number_of_samples(directory, target_var, total_channels):
    """
    Get number of npy samples per directory
    """
    files = os.listdir(directory)
    pred_str = '{}channels_{}_groundtruth'.format(total_channels, target_var)
    count = 0
    for f in files:
        if pred_str in f:
            count +=1

    return count


def groundtruth_array_flat(query_dir, num_channels, target):
    """
    Get groundtruth from model as array for plotting etc
    """
    channel_gt = []
    num_samples = get_number_of_samples(query_dir, target, num_channels)
    for i in range(int(num_samples)):
        arr = np.load('{}/{}channels_{}_groundtruth_{}.npy'.format(query_dir, num_channels, target, i))
        for j in range(int(arr.shape[0])):
            arr_flat = arr[j, 0, :, :].flatten()
            arr_list = arr_flat.tolist()
            channel_gt.extend(arr_list)

    groundtruth = np.array(channel_gt)
    return groundtruth


def pred_array_flat(query_dir, num_channels, target):
    """
    Get predictions from model as array for plotting etc
    """
    channel_preds = []
    num_samples = get_number_of_samples(query_dir, target, num_channels)
    for i in range(int(num_samples)):
        arr = np.load('{}/{}channels_{}_pred_{}.npy'.format(query_dir, num_channels, target, i))
        for j in range(int(arr.shape[0])):
            arr_flat = arr[j, 0, :, :].flatten()
            arr_list = arr_flat.tolist()
            channel_preds.extend(arr_list)

    preds = np.array(channel_preds)
    return preds


def get_pred_list(query_dir, num_channels, target):
    """
    Get predictions from model as list of 2d arrays for rmse
    """
    preds = []
    num_samples = get_number_of_samples(query_dir, target, num_channels)
    for i in range(int(num_samples)):
        arr = np.load('{}/{}channels_{}_pred_{}.npy'.format(query_dir, num_channels, target, i))
        for j in range(int(arr.shape[0])):
            preds.append(arr[j, 0, :, :])

    return preds


def get_groundtruth_list(query_dir, num_channels, target):
    """
    Get groundtruth from model as list of 2d arrays for rmse
    """
    gts = []
    num_samples = get_number_of_samples(query_dir, target, num_channels)
    for i in range(num_samples):
        arr = np.load('{}/{}channels_{}_groundtruth_{}.npy'.format(query_dir, num_channels, target, i))
        for j in range(int(arr.shape[0])):
            gts.append(arr[j, 0, :, :])

    return gts


def get_cqr_pred_list(query_dir, num_channels, target):
    """
    Get lower, upper calibrated predictions from model as lists of
    2d arrays for plotting
    """
    lower_preds = []
    upper_preds = []
    med_preds = []
    num_samples = get_number_of_samples(query_dir, target, num_channels)
    for i in range(num_samples):
        lower_arr = np.load('{}/{}channels_{}_pred_lower_cal_{}.npy'.format(query_dir, num_channels, target, i))
        upper_arr = np.load('{}/{}channels_{}_pred_upper_cal_{}.npy'.format(query_dir, num_channels, target, i))
        med_arr = np.load('{}/{}channels_{}_pred_med_cal_{}.npy'.format(query_dir, num_channels, target, i))
        for j in range(int(lower_arr.shape[0])):
            lower_preds.append(lower_arr[j][0, :, :])
            upper_preds.append(upper_arr[j][2, :, :])
            med_preds.append(med_arr[j][1, :, :])

    return lower_preds, upper_preds, med_preds


def get_cqr_length(lower_bound, upper_bound):
    """
    Get length of predictive interval from the upper and
    lower bound predictions
    """
    return np.abs(upper_bound-lower_bound)


def get_total_uncertainty_list(query_dir, num_channels, target):
    """
    Get total uncertainty from model as list of 2d arrays
    """
    uncs = []
    num_samples = get_number_of_samples(query_dir, target, num_channels)
    for i in range(int(num_samples)):
        arr = np.load('{}/{}channels_{}_total_pred_unc_maps_{}.npy'.format(query_dir, num_channels, target, i))
        for j in range(int(arr.shape[0])):
            uncs.append(arr[j, 0, :, :])

    return uncs


def get_ale_uncertainty_list(query_dir, num_channels, target):
    """
    Get total uncertainty from model as list of 2d arrays
    """
    uncs = []
    num_samples = get_number_of_samples(query_dir, target, num_channels)
    for i in range(int(num_samples)):
        arr = np.load('{}/{}channels_{}_ale_unc_maps_{}.npy'.format(query_dir, num_channels, target, i))
        for j in range(int(arr.shape[0])):
            uncs.append(arr[j, 0, :, :])

    return uncs

def get_epi_uncertainty_list(query_dir, num_channels, target):
    """
    Get total uncertainty from model as list of 2d arrays
    """
    uncs = []
    num_samples = get_number_of_samples(query_dir, target, num_channels)
    for i in range(int(num_samples)):
        arr = np.load('{}/{}channels_{}_epi_unc_maps_{}.npy'.format(query_dir, num_channels, target, i))
        for j in range(int(arr.shape[0])):
            uncs.append(arr[j, 0, :, :])

    return uncs


def get_nanmask_list(gt_list):
    """
    Get nan mask per groundtruth for spatial plotting
    """
    nan_list = []
    for i in range(len(gt_list)):
        nan_mask = np.isnan(gt_list[i])
        nan_mask_nans = np.where(nan_mask == True, np.nan, nan_mask)
        nan_mask_to_apply = np.where(nan_mask_nans == 0, 1, nan_mask_nans)
        nan_list.append(nan_mask_to_apply)

    return nan_list


def generate_plots(target, channel_list, region, target_dir):
    """
    Generate plots based on target
    """
    for channel in channel_list:
        for dir in os.listdir('{}/{}channels'.format(target_dir, channel)):
            print('Generating plots for run {}'.format(dir))
            if dir == '.DS_Store':
                continue
            query_directory = '{}/{}channels/{}'.format(target_dir, channel, dir)
            pred = pred_array_flat(query_directory, channel, target)
            groundtruth = groundtruth_array_flat(query_directory, channel, target)

            # Filter nans
            nan_mask = ~np.isnan(groundtruth)
            pred = pred[nan_mask]
            groundtruth = groundtruth[nan_mask]

            # Plot histogram
            plot_histogram(groundtruth, pred, query_directory, region, target)

            # Plot truth vs predictand
            truth_vs_predicted(groundtruth, pred, query_directory, region)


def calc_rmse(truth_list, predict_list, region, analysis_period):
    """
    Calculate and return rmse per point in sample
    :param: analysis period: june, july, august, summer
    """
    if analysis_period == 'june':
        num_samples = 30

    err_total = 0
    for i in range(len(truth_list)):
        err = truth_list[i] - predict_list[i]
        err_sq = err ** 2
        err_total = err_total + err_sq

    err_avg = err_total/num_samples
    rmse_arr = np.sqrt(err_avg)

    return rmse_arr

def calc_r2(truth_list, predict_list, nan_list, analysis_period):
    """
    Calculate r2 score
    """

    def remove_nans(arr, nan_arr):
        arr = arr.flatten()
        nan = nan_arr.flatten()
        arr = arr * nan
        arr = arr[~np.isnan(arr)]
        return arr

    r2_list = []
    for i in range(len(truth_list)):
        pred = remove_nans(predict_list[i], nan_list[i])
        truth = remove_nans(truth_list[i], nan_list[i])
        r2 = r2_score(truth, pred)
        r2_list.append(r2)

    return r2_list

def spatial_map(avg_data, target, metric, region, savedir):
    """
    Generate rmse map of results from rmse data
    """
    if region == 'NorthAmerica':
        # Hard coded vals
        bbox_extent = 'north_america'
        lon_vals = na_lon_vals
        lat_vals = na_lat_vals

    if region == 'Europe':
        bbox_extent = 'europe'
        lat_vals = eu_lat_vals
        lon_vals = eu_lon_vals

    if region == 'Globe':
        bbox_extent = 'globe'
        lat_vals = globe_lat_vals
        lon_vals = globe_lon_vals

    if target == 'mda8':
        if metric == 'rmse':
            vmin = 0
            vmax= 30
        elif metric == 'residual':
            vmin = -25
            vmax = 25
        elif metric == 'uncertainty':
            vmin = 0
            vmax = 90
        elif metric == 'aleatoric_uncertainty':
            vmin = 0
            vmax = 100
        elif metric == 'epistemic_uncertainty':
            vmin = 0
            vmax = 50
        else:
            vmin = 10
            vmax = 90

    if target == 'bias':
        if metric == 'rmse':
            vmin = 0
            vmax= 20
        elif metric == 'residual':
            vmin = -25
            vmax = 25
        elif metric == 'uncertainty':
            vmin = 0
            vmax = 100
        elif metric == 'aleatoric_uncertainty':
            vmin = 10
            vmax = 1000
        elif metric == 'epistemic_uncertainty':
            if region == 'Europe':
                vmax = 80
            if region == 'NorthAmerica':
                vmax = 200
            vmin = 10
        elif metric == 'upper_bound_predictions':
            vmin = 20
            vmax = 60
        elif metric == 'lower_bound_predictions':
            vmin = -10
            vmax = 20
        elif metric == 'avg_length':
            vmin = 30
            vmax = 50
        else:
            vmin = -5
            vmax = 50

    x, y = np.meshgrid(lon_vals, lat_vals, indexing='xy')
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    plt.pcolor(x, y, avg_data, cmap='coolwarm', vmin=vmin, vmax=vmax)
    ax.coastlines(linewidth=2)
    ax.set_facecolor('gray')
    #ax.stock_img()
    ax.set_extent(bbox_dict['{}'.format(bbox_extent)], crs=ccrs.PlateCarree())
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=1, color='gray', alpha=0.8, linestyle='--')
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    im_ratio = len(lat_vals) / len(lon_vals)
    cb = plt.colorbar(pad=0.1)
    plt.tight_layout()
    plt.title('{} {} for {}'.format(target, metric, region))
    plt.savefig('{}/{}_{}.png'.format(savedir, target, metric))
    plt.close()


def calculate_avg_2d_array(array_list, nan_masks_list, lat_count, lon_count):
    """
    Calculates the average over the test set samples
    per list of arrays specified
    """
    df = pd.DataFrame()
    for i in range(len(array_list)):
        arr = array_list[i] * nan_masks_list[i]
        arr_flat = arr.flatten()
        df['sample {}'.format(i)] = arr_flat

    avg_list = []
    for idx, row in df.iterrows():
        row_list = row.tolist()
        no_nans = [x for x in row_list if str(x) != 'nan']
        if len(no_nans) == 0:
            avg_list.append(np.nan)
        else:
            avg_list.append((sum(no_nans) / len(no_nans)))

    avg_arr = np.array(avg_list)
    avg_arr_2d = np.reshape(avg_arr, (lat_count, lon_count))

    return avg_arr_2d

def generate_timeseries_plot(data_dict, datelist, location, data_type, savedir):
    """
    Code to generate the plots themselves
    :param: data_dict: dictionary of data to plot
    :param: date_list: list of dates for labels
    :param: location: lat, lon point
    :param: data_type: grountruth, prediction interval
    """
    # Plot the locations that have data
    if len(datelist) > 31:
        spacing = 30
    else:
        spacing = 10

    fig = plt.figure(figsize=(12, 6))
    ax2 = fig.gca()
    plt.plot(datelist, data_dict)
    ax2.set_xticks(np.arange(0, len(datelist) + 1, spacing))
    plt.title('Timeseries of {} for location: {}'.format(location, data_type))
    plt.savefig('{}/Timeseries_{}_{}.png'.format(savedir, data_type, location))
    plt.close()


def timeseries_plots(final_dict, analysis_period, data_type, savedir):
    """
    Generate timeseries plots per location for test set
    """
    start, end = None, None
    if analysis_period == 'june':
        start = date(2019,6,1)
        end = date(2019,6,30)
    elif analysis_period == 'july':
        start = date(2019, 7, 1)
        end = date(2019, 7, 31)
    elif analysis_period == 'aug':
        start = date(2019, 8, 1)
        end = date(2019, 8, 31)
    elif analysis_period == 'summer':
        start = date(2019, 6, 1)
        end = date(2019, 8, 31)

    datelist = daterange(start, end)

    # Getting avg length or total unc values to print out the TOAR station that had the largest and smallest
    if data_type in ['interval_length', 'total_unc']:
        avg_list = []
        for k in final_dict.keys():
            avg_list.append(np.mean(final_dict[k]))
        # Get list of keys
        key_list = list(final_dict.keys())
        max_loc = avg_list.index(np.nanmax(avg_list))
        min_loc = avg_list.index(np.nanmin(avg_list))
        with open("{}/max_min_locs.txt".format(save_dir), "a") as f:
            print("TOAR location with max avg {} is: {}".format(data_type, key_list[max_loc]), file=f)
            print("TOAR location with min avg {} is: {}".format(data_type, key_list[min_loc]), file=f)

    for k in final_dict.keys():
        generate_timeseries_plot(final_dict[k], datelist, k, data_type, savedir)


def plot_max_min_timeseries(length_dict, point_pred, gt, upper, lower, savedir):
    """
    Plot cqr timeseries for max and min locations
    """
    avg_list = []
    for k in length_dict.keys():
        avg_list.append(np.mean(length_dict[k]))
    # Get list of keys
    key_list = list(length_dict.keys())
    max_loc = avg_list.index(np.nanmax(avg_list))
    min_loc = avg_list.index(np.nanmin(avg_list))

    max_key = key_list[max_loc]
    min_key = key_list[min_loc]
    start = date(2019, 6, 1)
    end = date(2019, 6, 30)
    datelist = daterange(start, end)

    max_length = length_dict[max_key]
    max_point = point_pred[max_key]
    max_gt = gt[max_key]
    max_upper = upper[max_key]
    max_lower = lower[max_key]

    min_length = length_dict[min_key]
    min_point = point_pred[min_key]
    min_gt = gt[min_key]
    min_upper = upper[min_key]
    min_lower = lower[min_key]

    import ipdb
    ipdb.set_trace()

    fig, ax = plt.subplots()
    ax.plot(datelist, max_gt, '-', label='GroundTruth')
    ax.fill_between(datelist, max_lower, max_upper, alpha=0.2)
    ax.set_xticks(np.arange(0, len(datelist) + 1, 9))
    ax.plot(datelist, max_point, 'o', label='Prediction')
    plt.title('Groundtruth and Predictions for Maximum Avg Interval Length at: {}'.format(max_key))
    plt.xlabel('DOY')
    plt.ylabel('Bias')
    ax.set_ylim(-10, 80)
    ax.legend()
    plt.savefig('{}/Max_Interval_Timeseries_{}.png'.format(savedir, max_key))
    plt.close()

    fig, ax = plt.subplots()
    ax.plot(datelist, min_gt, '-', label='GroundTruth')
    ax.fill_between(datelist, min_lower, min_upper, alpha=0.2)
    ax.set_xticks(np.arange(0, len(datelist) + 1, 9))
    ax.plot(datelist, min_point, 'o', label='Prediction')
    plt.title('Groundtruth and Predictions for Min Avg Interval Length at: {}'.format(min_key))
    plt.xlabel('DOY')
    plt.ylabel('Bias')
    ax.set_ylim(-10, 80)
    ax.legend()
    plt.savefig('{}/Min_Interval_Timeseries_{}.png'.format(savedir, min_key))
    plt.close()


def generate_loc_dict(arrays, region, analysis_period, data_type, savedir):
    """
    Generate dictionary of data per location

    """
    start, end = None, None
    if analysis_period == 'june':
        start = date(2019, 6, 1)
        end = date(2019, 6, 30)
    elif analysis_period == 'july':
        start = date(2019, 7, 1)
        end = date(2019, 7, 31)
    elif analysis_period == 'aug':
        start = date(2019, 8, 1)
        end = date(2019, 8, 31)
    elif analysis_period == 'summer':
        start = date(2019, 6, 1)
        end = date(2019, 8, 31)

    datelist = daterange(start, end)

    if region == 'NorthAmerica':
        lon_vals = na_lon_vals
        lat_vals = na_lat_vals

    if region == 'Europe':
        lat_vals = eu_lat_vals
        lon_vals = eu_lon_vals

    if region == 'Globe':
        lat_vals = globe_lat_vals
        lon_vals = globe_lon_vals

    timeseries_dict = {}
    for i in range(len(arrays)):
        doy = datelist[i]
        arr = arrays[i]
        timeseries_dict['{}'.format(doy)] = arr.flatten().tolist()

    # iterate through the lat and lon vals and plot
    toar_dict = {}
    for lat in lat_vals:
        for lon in lon_vals:
            toar_dict['lat:{},lon:{}'.format(lat, lon)] = []

    timeseries_dict_keys_list = list(timeseries_dict.keys())
    i = 0
    for k, v in toar_dict.items():
        for day in timeseries_dict_keys_list:
            toar_dict[k].append(timeseries_dict[day][i])
        i += 1

    keys_to_drop = []
    for k, v in toar_dict.items():
        if np.isnan(toar_dict[k]).all():
            print('Dropping: {}'.format(k))
            keys_to_drop.append(k)

    final_dict = {key: value for key, value in toar_dict.items() if key not in keys_to_drop}
    # Save data for further plotting
    with open('{}/{}_timeseries_data.pickle'.format(savedir,data_type), 'wb') as handle:
        pickle.dump(final_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return final_dict

def plot_r2_timeseries(r2_list, analysis_period, region, save_dir):
    """
    Generates plots for r2 timeseries
    """
    start, end = None, None
    if analysis_period == 'june':
        start = date(2019, 6, 1)
        end = date(2019, 6, 30)
    elif analysis_period == 'july':
        start = date(2019, 7, 1)
        end = date(2019, 7, 31)
    elif analysis_period == 'aug':
        start = date(2019, 8, 1)
        end = date(2019, 8, 31)
    elif analysis_period == 'summer':
        start = date(2019, 6, 1)
        end = date(2019, 8, 31)

    datelist = daterange(start, end)

    ig, ax = plt.subplots()
    ax.plot(datelist, r2_list, '-', label='R2 Score')
    ax.set_xticks(np.arange(0, len(datelist) + 1, 9))
    plt.title('R^2 Score per Day for {}'.format(region))
    plt.xlabel('DOY')
    plt.ylabel('R^2')
    plt.savefig('{}/R2_score_timeseries.png'.format(save_dir))
    plt.close()


def generate_standard_plots(channels, target, num_lats, num_lons, region, save_dir, analysis_period):
    """
    Generate appropriate plots for Standard model
    """
    pred_list = get_pred_list(save_dir, channels, target)
    groundtruth_list = get_groundtruth_list(save_dir, channels, target)

    # Get nan masks for plotting
    nan_mask_list = get_nanmask_list(groundtruth_list)

    # Get avg and plot groundtruth
    gt_avg_arr_2d = calculate_avg_2d_array(groundtruth_list, nan_mask_list, num_lats, num_lons)
    np.save('{}/avg_groundtruth.npy'.format(save_dir), gt_avg_arr_2d)
    spatial_map(gt_avg_arr_2d, target, 'groundtruth', region, save_dir)

    # Calc avg pred and plot
    pred_avg_arr_2d = calculate_avg_2d_array(pred_list, nan_mask_list, num_lats, num_lons)
    np.save('{}/avg_prediction.npy'.format(save_dir), pred_avg_arr_2d)
    spatial_map(pred_avg_arr_2d, target, 'predictions', region, save_dir)

    # Get avg residual and plot
    avg_residual = pred_avg_arr_2d - gt_avg_arr_2d
    avg_residual = avg_residual.astype('float32')
    np.save('{}/avg_residual.npy'.format(save_dir), avg_residual)
    spatial_map(avg_residual, target, 'residual', region, save_dir)

    # Get avg rmse and plot
    rmse_calc = calc_rmse(groundtruth_list, pred_list, region, analysis_period)
    np.save('{}/avg_rmse.npy'.format(save_dir), rmse_calc)
    spatial_map(rmse_calc, target, 'rmse', region, save_dir)


    pred_list_nans = []
    for i in range(len(nan_mask_list)):
        pred_list_nans.append(pred_list[i])

    groundtruth_dict = generate_loc_dict(groundtruth_list, region, analysis_period, 'groundtruth', save_dir)
    pred_dict = generate_loc_dict(pred_list_nans, region, analysis_period, 'point_predictions', save_dir)

    # Generate timeseries plots per location
    timeseries_plots(pred_dict, analysis_period, 'point_predictions', save_dir)
    timeseries_plots(groundtruth_dict, analysis_period, 'groundtruth', save_dir)



def generate_mcdropout_plots(channels, target, num_lats, num_lons, region, save_dir, analysis_period):
    """
    Generate appropriate plots for MCDropout model
    """
    pred_list = get_pred_list(save_dir, channels, target)
    groundtruth_list = get_groundtruth_list(save_dir, channels, target)
    total_uncertainty_list = get_total_uncertainty_list(save_dir, channels, target)
    epi_uncertainty_list = get_epi_uncertainty_list(save_dir, channels, target)
    ale_uncertainty_list = get_ale_uncertainty_list(save_dir, channels, target)

    # Get nan masks for plotting
    nan_mask_list = get_nanmask_list(groundtruth_list)

    # Get avg and plot groundtruth
    gt_avg_arr_2d = calculate_avg_2d_array(groundtruth_list, nan_mask_list, num_lats, num_lons)
    np.save('{}/avg_groundtruth.npy'.format(save_dir), gt_avg_arr_2d)
    spatial_map(gt_avg_arr_2d, target, 'groundtruth', region, save_dir)

    # Calc avg pred and plot
    pred_avg_arr_2d = calculate_avg_2d_array(pred_list, nan_mask_list, num_lats, num_lons)
    np.save('{}/avg_prediction.npy'.format(save_dir), pred_avg_arr_2d)
    spatial_map(pred_avg_arr_2d, target, 'predictions', region, save_dir)

    # Get avg residual and plot
    avg_residual = pred_avg_arr_2d - gt_avg_arr_2d
    avg_residual = avg_residual.astype('float32')
    np.save('{}/avg_residual.npy'.format(save_dir), avg_residual)
    spatial_map(avg_residual, target, 'residual', region, save_dir)

    # Get avg rmse and plot
    rmse_calc = calc_rmse(groundtruth_list, pred_list, region, analysis_period)
    np.save('{}/avg_rmse.npy'.format(save_dir), rmse_calc)
    spatial_map(rmse_calc, target, 'rmse', region, save_dir)

    # Get avg predictive uncertainty and plot
    unc_avg_arr_2d = calculate_avg_2d_array(total_uncertainty_list, nan_mask_list, num_lats, num_lons)
    np.save('{}/avg_total_uncertainty.npy'.format(save_dir), unc_avg_arr_2d)
    spatial_map(unc_avg_arr_2d, target, 'uncertainty', region, save_dir)

    # Get avg aleatoric uncertainty and plot
    unc_avg_ale_arr_2d = calculate_avg_2d_array(ale_uncertainty_list, nan_mask_list, num_lats, num_lons)
    np.save('{}/avg_aleatoric_uncertainty.npy'.format(save_dir), unc_avg_ale_arr_2d)
    spatial_map(unc_avg_ale_arr_2d, target, 'aleatoric_uncertainty', region, save_dir)

    # Get avg epistemic uncertainty and plot
    unc_avg_epi_arr_2d = calculate_avg_2d_array(epi_uncertainty_list, nan_mask_list, num_lats, num_lons)
    np.save('{}/avg_epistemic_uncertainty.npy'.format(save_dir), unc_avg_epi_arr_2d)
    spatial_map(unc_avg_epi_arr_2d, target, 'epistemic_uncertainty', region, save_dir)

    # Apply nan mask and get timeseries
    ale_list_nans = []
    epi_list_nans = []
    total_unc_list_nans = []
    pred_list_nans = []
    for i in range(len(nan_mask_list)):
        ale_list_nans.append(ale_uncertainty_list [i] * nan_mask_list[i])
        epi_list_nans.append(epi_uncertainty_list[i] * nan_mask_list[i])
        total_unc_list_nans.append(total_uncertainty_list[i] * nan_mask_list[i])
        pred_list_nans.append(pred_list[i])

    ale_dict = generate_loc_dict(ale_list_nans, region, analysis_period, 'aleatoric', save_dir)
    epi_dict = generate_loc_dict(epi_list_nans, region, analysis_period, 'epistemic', save_dir)
    total_unc_dict = generate_loc_dict(total_unc_list_nans, region, analysis_period, 'total_unc', save_dir)
    groundtruth_dict = generate_loc_dict(groundtruth_list, region, analysis_period, 'groundtruth', save_dir)
    pred_dict = generate_loc_dict(pred_list_nans, region, analysis_period, 'point_predictions', save_dir)

    # Generate timeseries plots per location
    timeseries_plots(ale_dict, analysis_period, 'aleatoric', save_dir)
    timeseries_plots(epi_dict, analysis_period, 'epistemic', save_dir)
    timeseries_plots(total_unc_dict, analysis_period, 'total_unc', save_dir)
    timeseries_plots(pred_dict, analysis_period, 'point_predictions', save_dir)
    timeseries_plots(groundtruth_dict, analysis_period, 'groundtruth', save_dir)


def generate_cqr_plots(channels, target, num_lats, num_lons, region, save_dir, analysis_period):
    """
    Generate appropriate plots for CQR model
    """

    groundtruth_list = get_groundtruth_list(save_dir, channels, target)
    lower_bound_pred_list, upper_bound_pred_list, med_pred_list = get_cqr_pred_list(save_dir, channels, target)

    # Get nan masks for plotting
    nan_mask_list = get_nanmask_list(groundtruth_list)

    # Get avg and plot groundtruth
    gt_avg_arr_2d = calculate_avg_2d_array(groundtruth_list, nan_mask_list, num_lats, num_lons)
    np.save('{}/avg_groundtruth.npy'.format(save_dir), gt_avg_arr_2d)
    spatial_map(gt_avg_arr_2d, target, 'groundtruth', region, save_dir)

    # Calc avg pred and plot
    pred_avg_arr_2d_lower = calculate_avg_2d_array(lower_bound_pred_list, nan_mask_list, num_lats, num_lons)
    np.save('{}/avg_prediction_lower.npy'.format(save_dir), pred_avg_arr_2d_lower)
    pred_avg_arr_2d_upper = calculate_avg_2d_array(upper_bound_pred_list, nan_mask_list, num_lats, num_lons)
    np.save('{}/avg_prediction_upper.npy'.format(save_dir), pred_avg_arr_2d_upper)
    pred_avg_arr_2d_med = calculate_avg_2d_array(med_pred_list, nan_mask_list, num_lats, num_lons)
    np.save('{}/avg_prediction_med.npy'.format(save_dir), pred_avg_arr_2d_med)

    # Get avg rmse and plot
    rmse_calc = calc_rmse(groundtruth_list, med_pred_list, region, analysis_period)
    np.save('{}/avg_rmse.npy'.format(save_dir), rmse_calc)

    spatial_map(rmse_calc, target, 'rmse', region, save_dir)

    # Get r2 score and plot
    r2_timeseries = calc_r2(groundtruth_list, med_pred_list, nan_mask_list, region)
    np.save('{}/r2_timeseries.npy'.format(save_dir), r2_timeseries)
    plot_r2_timeseries(r2_timeseries, analysis_period, region, save_dir)

    spatial_map(pred_avg_arr_2d_lower, target, 'lower_bound_predictions', region, save_dir)
    spatial_map(pred_avg_arr_2d_upper, target, 'upper_bound_predictions', region, save_dir)
    spatial_map(pred_avg_arr_2d_med, target, 'point_predictions', region, save_dir)

    # Calc avg bound length
    avg_2d_length = get_cqr_length(pred_avg_arr_2d_lower, pred_avg_arr_2d_upper)
    np.save('{}/avg_length.npy'.format(save_dir), avg_2d_length)
    spatial_map(avg_2d_length, target, 'avg_length', region, save_dir)

    # Apply nan mask and plot timeseries
    lower_bound_pred_list_nans = []
    upper_bound_pred_list_nans = []
    med_pred_list_nans = []
    interval_length = []
    for i in range(len(nan_mask_list)):
        lower_bound_pred_list_nans.append(lower_bound_pred_list[i] * nan_mask_list[i])
        upper_bound_pred_list_nans.append(upper_bound_pred_list[i] * nan_mask_list[i])
        med_pred_list_nans.append(med_pred_list[i] * nan_mask_list[i])
        interval_length.append(upper_bound_pred_list_nans[i]-lower_bound_pred_list_nans[i])

    lower_bound_dict = generate_loc_dict(lower_bound_pred_list_nans, region, analysis_period, 'lower_bound', save_dir)
    upper_bound_dict = generate_loc_dict(upper_bound_pred_list_nans, region, analysis_period, 'upper_bound', save_dir)
    med_dict = generate_loc_dict(med_pred_list_nans, region, analysis_period, 'point_predictions', save_dir)
    groundtruth_dict = generate_loc_dict(groundtruth_list, region, analysis_period, 'groundtruth', save_dir)
    interval_length_dict = generate_loc_dict(interval_length, region, analysis_period, 'interval_length', save_dir)

    # Generate timeseries plots per location
    #timeseries_plots(lower_bound_dict, analysis_period, 'lower_bound', save_dir)
    #timeseries_plots(upper_bound_dict, analysis_period, 'upper_bound', save_dir)
    #timeseries_plots(med_dict, analysis_period, 'point_predictions', save_dir)
    #timeseries_plots(groundtruth_dict, analysis_period, 'groundtruth', save_dir)
    timeseries_plots(interval_length_dict, analysis_period, 'interval_length', save_dir)

    # Plot max and min timeseries
    plot_max_min_timeseries(interval_length_dict,med_dict,groundtruth_dict,upper_bound_dict,lower_bound_dict, save_dir)

def calc_avg_from_file(metric, num_lats, num_lons, directory, model):
    """
    Calculate avg of array from saved files of
    individual runs
    :param: metric: metric to calculate mean for
    """
    files = os.listdir(directory)
    if model == 'cqr_avg':
        if metric != 'groundtruth':
            pred_str = 'avg_prediction_{}'.format(metric)
        if metric == 'groundtruth':
            pred_str = 'avg_groundtruth'
        if metric == 'length':
            pred_str = 'length'
        if metric == 'rmse':
            pred_str = 'rmse'
        if metric == 'r2':
            pred_str = 'r2_timeseries'
        arr_list = []
        for f in files:
            if pred_str in f:
                arr = np.load('{}/{}'.format(directory, f))
                arr_list.append(arr)

    if model == 'mcdropout_avg':
        pred_str = 'avg_{}'.format(metric)
        if metric == 'r2':
            pred_str = 'r2_timeseries'
        arr_list = []
        for f in files:
            if pred_str in f:
                arr_list.append(np.load('{}/{}'.format(directory, f)))

    if metric == 'r2':
        r2_avg = np.mean(arr_list, axis=0)
        return r2_avg

    # --- Getting nan_mask_list
    groundtruth_list = []
    for f in files:
        if 'groundtruth' in f:
            arr = np.load('{}/{}'.format(directory, f))
            groundtruth_list.append(arr)

    nan_mask_list = get_nanmask_list(groundtruth_list)

    # Get avg and plot groundtruth
    avg_arr_2d = calculate_avg_2d_array(arr_list, nan_mask_list, num_lats, num_lons)

    return avg_arr_2d


def generate_avg_run_plots(target, region, num_lats, num_lons, save_dir, model):
    """
    Generate plots from multiple runs to obtain average to report
    """
    if model == 'cqr_avg':
        # --- Load and get mean of data
        gt = calc_avg_from_file('groundtruth', num_lats, num_lons, save_dir, model)
        lower = calc_avg_from_file('lower', num_lats, num_lons, save_dir, model)
        upper = calc_avg_from_file('upper', num_lats, num_lons, save_dir, model)
        med = calc_avg_from_file('med', num_lats, num_lons, save_dir, model)
        length = calc_avg_from_file('length', num_lats, num_lons, save_dir, model)
        rmse = calc_avg_from_file('rmse', num_lats, num_lons, save_dir, model)
        #r2 = calc_avg_from_file('r2', num_lats, num_lons, save_dir, model)

        # --- Generate spatial maps
        spatial_map(gt, target, 'groundtruth', region, save_dir)
        spatial_map(lower, target, 'lower_bound_predictions', region, save_dir)
        spatial_map(upper, target, 'upper_bound_predictions', region, save_dir)
        spatial_map(med, target, 'point_predictions', region, save_dir)
        spatial_map(length, target, 'avg_length', region, save_dir)
        spatial_map(rmse, target, 'rmse', region, save_dir)

        # --- Generate time series map
        #plot_r2_timeseries(r2, analysis_period, region, save_dir)

        # --- Get overall min/max metrics
        with open("{}/avg_scores.txt".format(save_dir), "a") as f:
            print("Max average interval length is: {}".format(np.nanmax(length)), file=f)
            print("Min average interval length is: {}".format(np.nanmin(length)), file=f)
            print("Average interval length is: {}".format(np.nanmean(length)), file=f)
            print("Variance of interval length is: {}".format(np.nanvar(length)), file=f)

    if model == 'mcdropout_avg':
        # --- Load and get mean of data
        gt = calc_avg_from_file('groundtruth', num_lats, num_lons, save_dir, model)
        pred = calc_avg_from_file('prediction', num_lats, num_lons, save_dir, model)
        total_unc = calc_avg_from_file('total_uncertainty', num_lats, num_lons, save_dir, model)
        ale_unc = calc_avg_from_file('aleatoric_uncertainty', num_lats, num_lons, save_dir, model)
        epi_unc = calc_avg_from_file('epistemic_uncertainty', num_lats, num_lons, save_dir, model)
        rmse = calc_avg_from_file('rmse', num_lats, num_lons, save_dir, model)
        r2 = calc_avg_from_file('r2', num_lats, num_lons, save_dir, model)

        # --- Generate spatial maps
        spatial_map(gt, target, 'groundtruth', region, save_dir)
        spatial_map(pred, target, 'prediction', region, save_dir)
        spatial_map(total_unc, target, 'total_uncertainty', region, save_dir)
        spatial_map(ale_unc, target, 'aleatoric_uncertainty', region, save_dir)
        spatial_map(epi_unc, target, 'epistemic_uncertainty', region, save_dir)
        spatial_map(rmse, target, 'rmse', region, save_dir)

        # --- Generate time series map
        #plot_r2_timeseries(r2, analysis_period, region, save_dir)

        # --- Get overall min/max metrics
        with open("{}/avg_scores.txt".format(save_dir), "a") as f:
            print("Max average epi is: {}".format(np.nanmax(epi_unc)), file=f)
            print("Min average epi is: {}".format(np.nanmin(epi_unc)), file=f)
            print("Average epi is: {}".format(np.nanmean(epi_unc)), file=f)
            print("Variance of epi is: {}".format(np.nanvar(epi_unc)), file=f)


def generate_rfplots(channels, target, num_lats, num_lons, region, save_dir):
    """
    Generate appropriate plots for Standard model
    """
    preds = xr.open_dataset('{}/june.test.quantiles.nc'.format(save_dir))
    # Hardcoding date list for 2019
    date_list = daterange(date(2019, 6, 1), date(2019, 6, 30))
    pred_list = []
    lower_pred = []
    upper_pred = []
    for d in date_list:
        rf_ds = preds.sel(time=d)['0.5']
        rf_lower = preds.sel(time=d)['0.05']
        rf_higher = preds.sel(time=d)['0.95']
        pred_list.append(rf_ds)
        lower_pred.append(rf_lower)
        upper_pred.append(rf_higher)

    groundtruth_list = get_groundtruth_list(save_dir, channels, target)
    nan_mask_list = get_nanmask_list(groundtruth_list)

    avg_2d_length = get_cqr_length(np.array(lower_pred), np.array(upper_pred))
    np.save('{}/avg_length.npy'.format(save_dir), avg_2d_length)

    # Get avg rmse and plot
    rmse_calc = calc_rmse(groundtruth_list, pred_list, region, analysis_period)
    np.save('{}/avg_rmse.npy'.format(save_dir), rmse_calc)

    with open("{}/avg_scores.txt".format(save_dir), "a") as f:
        print("Avg rmse is: {}".format(np.nanmean(rmse_calc)), file=f)
        print("Avg length is: {}".format(np.nanmean(avg_2d_length)), file=f)
        print("Max average interval length is: {}".format(np.nanmax(avg_2d_length)), file=f)
        print("Min average interval length is: {}".format(np.nanmin(avg_2d_length)), file=f)


if __name__ == '__main__':
    args = get_args()
    channels = args.channels
    region= args.region
    save_dir = args.save_dir
    target = args.target
    model = args.model
    analysis_period = args.analysis_period
    sensitivity = args.sensitivity

    if sensitivity:
        channels = int(channels) -1
    else:
        channels = channels

    if region == 'Europe':
        num_lats = 27
        num_lons = 31
    if region == 'NorthAmerica':
        num_lats = 31
        num_lons = 49
    if region == 'Globe':
        num_lats = 160
        num_lons = 320

    if model == 'standard':
        generate_standard_plots(channels, target, num_lats, num_lons, region, save_dir, analysis_period)

    if model == 'mcdropout':
        generate_mcdropout_plots(channels, target, num_lats, num_lons, region, save_dir, analysis_period)

    if model == 'cqr':
        generate_cqr_plots(channels, target, num_lats, num_lons, region, save_dir, analysis_period)

    if model in ['cqr_avg', 'mcdropout_avg']:
        generate_avg_run_plots(target, region, num_lats, num_lons, save_dir, model)

    if model == 'rf':
        generate_rfplots(channels, target, num_lats, num_lons, region, save_dir)
