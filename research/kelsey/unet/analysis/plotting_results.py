"""
Script for plotting different results for nicer viz
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import math
import xarray as xr
from tqdm import tqdm
import pandas as pd
from scipy import stats
from scipy.stats import ks_2samp        # use to quantify the difference of two distributions
import argparse
import math
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from matplotlib import colors
from numpy import moveaxis

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
    pred_str = '{}channels_{}_pred'.format(total_channels, target_var)
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
    for i in range(int(num_samples)):
        arr = np.load('{}/{}channels_{}_groundtruth_{}.npy'.format(query_dir, num_channels, target, i))
        for j in range(int(arr.shape[0])):
            gts.append(arr[j, 0, :, :])

    return gts

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

def calc_rmse(truth, predict, region):
    """
    Calculate and return rmse per point in sample
    """
    if region == 'NorthAmerica':
        list_step = 49
    if region == 'Europe':
        list_step = 31

    truth = np.hstack(truth)
    predict = np.hstack(predict)

    rmse_list = []

    for i in range(len(truth)):
        if np.isnan(truth[i]):
            rmse_list.append(np.nan)
        else:
            mse = np.square(np.subtract(truth[i], predict[i])).mean()
            rmse = math.sqrt(mse)
            rmse_list.append(rmse)

    total_list = []
    for i in range(0, len(rmse_list), list_step):
        dummy_list = [rmse_list[i:i + list_step]]
        total_list.append(dummy_list)

    arr = np.array(total_list)
    arr = np.moveaxis(arr, 0,1)
    rmse_arr = arr[0,:,:]

    return rmse_arr

def spatial_map(avg_data, target, metric, region, savedir):
    """
    Generate rmse map of results from rmse data
    """
    if region == 'NorthAmerica':
        # Hard coded vals from NA extent
        bbox_extent = 'north_america'
        lon_vals = [-124.875, -123.75 , -122.625, -121.5  , -120.375, -119.25 ,
           -118.125, -117.   , -115.875, -114.75 , -113.625, -112.5  ,
           -111.375, -110.25 , -109.125, -108.   , -106.875, -105.75 ,
           -104.625, -103.5  , -102.375, -101.25 , -100.125,  -99.   ,
            -97.875,  -96.75 ,  -95.625,  -94.5  ,  -93.375,  -92.25 ,
            -91.125,  -90.   ,  -88.875,  -87.75 ,  -86.625,  -85.5  ,
            -84.375,  -83.25 ,  -82.125,  -81.   ,  -79.875,  -78.75 ,
            -77.625,  -76.5  ,  -75.375,  -74.25 ,  -73.125,  -72.   ,
            -70.875]

        lat_vals = [20.748, 21.869, 22.991, 24.112, 25.234, 26.355, 27.476, 28.598,
           29.719, 30.841, 31.962, 33.084, 34.205, 35.327, 36.448, 37.57 ,
           38.691, 39.813, 40.934, 42.056, 43.177, 44.299, 45.42 , 46.542,
           47.663, 48.785, 49.906, 51.028, 52.149, 53.271, 54.392]

    if region == 'Europe':
        bbox_extent = 'europe'
        lat_vals = [35.327, 36.448, 37.57, 38.691, 39.813, 40.934, 42.056, 43.177,
                    44.299, 45.42, 46.542, 47.663, 48.785, 49.906, 51.028, 52.149,
                    53.271, 54.392, 55.514, 56.635, 57.757, 58.878, 60., 61.121, 62.242, 63.364, 64.485]

        lon_vals = [-9., -7.875, -6.75, -5.625, -4.5, -3.375, -2.25, -1.125, 0., 1.125,
                    2.25, 3.375, 4.5, 5.625, 6.75, 7.875, 9., 10.125, 11.25, 12.375, 13.5,
                    14.625, 15.75, 16.875, 18., 19.125, 20.25, 21.375, 22.5, 23.625, 24.75]

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
            vmax = 50
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
            vmax = 50
        else:
            vmin = -20
            vmax = 50


    x, y = np.meshgrid(lon_vals, lat_vals, indexing='xy')
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    plt.pcolor(x, y, avg_data, cmap='coolwarm', vmin=vmin, vmax=vmax)
    ax.coastlines()
    #ax.stock_img()
    ax.set_extent(bbox_dict['{}'.format(bbox_extent)], crs=ccrs.PlateCarree())  # NA region
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    im_ratio = len(lat_vals) / len(lon_vals)
    cb = plt.colorbar(pad=0.1)
    plt.tight_layout()
    plt.title('{} {} for {}'.format(target, metric, region))
    #plt.show()
    plt.savefig('{}/{}_{}.png'.format(savedir, target, metric))
    plt.close()


def calculate_avg_2d_array(array_list, nan_masks_list, lat_count, lon_count):
    """
    Calculates the average over the test set samples
    per list of arrays specified
    """
    # Get avg predictive uncertainty and plot
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

channels = [39]
region= 'NorthAmerica'
targets = ['bias']

target_dir = '/Users/kelseydoerksen/Desktop/unet/runs/{}/{}'.format(region, targets[0])
# Run results
generate_plots(targets[0], channels, region, target_dir)

if region == 'Europe':
    num_lats = 27
    num_lons = 31
if region == 'NorthAmerica':
    num_lats = 31
    num_lons = 49

for target in targets:
    target_dir = '/Users/kelseydoerksen/Desktop/unet/runs/{}/{}'.format(region, target)   # To update this to be more dynamic
    for channel in channels:
        for dir in os.listdir('{}/{}channels'.format(target_dir, channel)):
            print('Calculating rmse for run {}'.format(dir))
            if dir == '.DS_Store':
                continue
            print('Calculating metrics for run {}'.format(dir))
            query_directory = '{}/{}channels/{}'.format(target_dir, channel, dir)

            pred_list = get_pred_list(query_directory, channel, target)
            groundtruth_list = get_groundtruth_list(query_directory, channel, target)
            #total_uncertainty_list = get_total_uncertainty_list(query_directory, channel, target)
            #epi_uncertainty_list = get_epi_uncertainty_list(query_directory, channel, target)
            #ale_uncertainty_list = get_ale_uncertainty_list(query_directory, channel, target)

            # Get nan masks for plotting
            nan_mask_list = get_nanmask_list(groundtruth_list)

            # Get avg and plot groundtruth
            gt_avg_arr_2d = calculate_avg_2d_array(groundtruth_list, nan_mask_list, num_lats, num_lons)
            spatial_map(gt_avg_arr_2d, target, 'groundtruth', region, query_directory)

            # Calc avg pred and plot
            pred_avg_arr_2d = calculate_avg_2d_array(pred_list, nan_mask_list, num_lats, num_lons)
            spatial_map(pred_avg_arr_2d, target, 'predictions', region, query_directory)

            # Get avg residual and plot
            avg_residual = pred_avg_arr_2d - gt_avg_arr_2d
            avg_residual = avg_residual.astype('float32')
            spatial_map(avg_residual, target, 'residual', region, query_directory)

            # Get avg rmse and plot
            rmse_calc = calc_rmse(gt_avg_arr_2d, pred_avg_arr_2d, region)
            spatial_map(rmse_calc, target, 'rmse', region, query_directory)

            '''
            # Get avg predictive uncertainty and plot
            unc_avg_arr_2d = calculate_avg_2d_array(total_uncertainty_list, nan_mask_list, num_lats, num_lons)
            spatial_map(unc_avg_arr_2d, target, 'uncertainty', region, query_directory)

            # Get avg aleatoric uncertainty and plot
            unc_avg_arr_2d = calculate_avg_2d_array(ale_uncertainty_list, nan_mask_list, num_lats, num_lons)
            spatial_map(unc_avg_arr_2d, target, 'aleatoric_uncertainty', region, query_directory)

            # Get avg epistemic uncertainty and plot
            unc_avg_arr_2d = calculate_avg_2d_array(epi_uncertainty_list, nan_mask_list, num_lats, num_lons)
            spatial_map(unc_avg_arr_2d, target, 'epistemic_uncertainty', region, query_directory)
            '''