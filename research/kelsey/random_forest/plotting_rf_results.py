"""
Script to process the results and plot from RF
"""

import numpy as np
import os
import xarray as xr
from tqdm import tqdm
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import argparse
import math
from sklearn.metrics    import (
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score
)
from joblib import dump
from sklearn.inspection import permutation_importance
from scipy.stats import pearsonr
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
            'north america': [-140, -50, 10, 80],
            'west_europe': [-20, 10, 25, 80],
            'east_europe': [10, 40, 25, 80],
            'west_na': [-140, -95, 10, 80],
            'east_na': [-95, -50, 10, 80],
            'east_europe1': [20, 35, 40, 50]}

def calc_importances(model, feature_names, dir):
    '''
    Calculate feature importances, save as txt
    and plot
    '''
    importances = model.feature_importances_
    stddev = np.std([est.feature_importances_ for est in model.estimators_], axis=0)
    df = pd.DataFrame(np.array([importances, stddev]), columns=feature_names, index=['importance', 'stddev'])
    df = df.sort_values(by='importance', axis=1, ascending=False)
    fmt = {}
    for var, vals in df.items():
        fmt[var] = f'{vals.importance} +/- {vals.stddev}'

    '''
    Logger.info('Permutation importance +/- stddev:')
    strings = align_print(fmt, enum=True, print=Logger.info)
    with open('{}/importances.txt'.format(dir), 'w') as file:
        file.write('\n\nFeature Importance:\n')
        file.write('\n'.join(strings))
    '''

    return df


def plot_importances(imp, perm, dir, month, year, region):
    '''
    Generates bar plot for FI with perm
    '''

    # Normalize first
    imp = imp / imp.max(axis=1).importance
    perm = perm / perm.max(axis=1).importance

    X_axis = np.arange(20)
    plt.bar(X_axis - 0.2, imp.loc['importance'].values, yerr=imp.loc['stddev'].values,
            width=0.4, label='Importance')
    plt.bar(X_axis + 0.2, perm.loc['importance'].values, yerr=perm.loc['stddev'].values,
            width=0.4, label='Permutation Importance')
    plt.xticks(X_axis, imp, rotation=90)
    plt.title('Feature Importances for Testing set {} {} {}'.format(month, year, region))
    plt.xlabel('Feature Name')
    plt.ylabel('Importance')
    plt.legend()
    plt.tight_layout()
    plt.savefig('{}/importances.png'.format(dir))
    #plt.show()

def truth_vs_predicted(target, predict, dir, region):
    """
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # Retrieve the limits and expand them by 5% so everything fits into a square grid
    limits = min([target.min(), predict.min()]), max([target.max(), predict.max()])
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
    #plt.show()
    plt.close()


def plot_histogram(target, pred, dir, region, analysis_time):
    '''
    Plot histogram of true vs predicted
    '''

    bins = np.linspace(-120, 120, 300)
    plt.hist(target, bins, histtype='step', label=['target'])
    plt.hist(pred, bins, histtype='step', label=['prediction'])
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.xlabel('bias')
    plt.ylabel('count')

    if analysis_time in ['june', 'july', 'august']:
        plt.ylim(0, 700)
    else:
        plt.ylim(0, 9000)

    # save histogram data to plot with unet
    df = pd.DataFrame()
    df['gt'] = target
    df['pred'] = pred
    df.to_csv('{}/histogram_data.csv'.format(dir))

    plt.title('Truth vs Predicted Histogram for {}'.format(region))
    plt.savefig('{}/truth_vs_pred_hist.png'.format(dir))
    #plt.show()
    plt.close()

def plotting_spatial_data(avg_data, metric, model_target, anaylsis_date, save_directory, extent):
    """
    Plotting data now that we have the results setup nicely
    avg data: avg of either rmse, pred or label over lat, lon point
    metric: metric we are interested in plotting (rmse, prediction or groundtruth)
    analysis_date: month or season we are plotting
    """
    # Now we can plot finally
    na_lon_vals = [-124.875, -123.75, -122.625, -121.5, -120.375, -119.25,
                -118.125, -117., -115.875, -114.75, -113.625, -112.5,
                -111.375, -110.25, -109.125, -108., -106.875, -105.75,
                -104.625, -103.5, -102.375, -101.25, -100.125, -99.,
                -97.875, -96.75, -95.625, -94.5, -93.375, -92.25,
                -91.125, -90., -88.875, -87.75, -86.625, -85.5,
                -84.375, -83.25, -82.125, -81., -79.875, -78.75,
                -77.625, -76.5, -75.375, -74.25, -73.125, -72.,
                -70.875]

    na_lat_vals = [20.748, 21.869, 22.991, 24.112, 25.234, 26.355, 27.476, 28.598,
                29.719, 30.841, 31.962, 33.084, 34.205, 35.327, 36.448, 37.57,
                38.691, 39.813, 40.934, 42.056, 43.177, 44.299, 45.42, 46.542,
                47.663, 48.785, 49.906, 51.028, 52.149, 53.271, 54.392]

    eu_lat_vals = [35.327, 36.448, 37.57, 38.691, 39.813, 40.934, 42.056, 43.177,
               44.299, 45.42, 46.542, 47.663, 48.785, 49.906, 51.028, 52.149,
               53.271, 54.392, 55.514, 56.635, 57.757, 58.878, 60., 61.121,
               62.242, 63.364, 64.485]
    eu_lon_vals = [-9., -7.875, -6.75, -5.625, -4.5, -3.375, -2.25, -1.125,
               0., 1.125, 2.25, 3.375, 4.5, 5.625, 6.75, 7.875,
               9., 10.125, 11.25, 12.375, 13.5, 14.625, 15.75, 16.875,
               18., 19.125, 20.25, 21.375, 22.5, 23.625, 24.75]

    if extent == 'NorthAmerica':
        lat_vals = na_lat_vals
        lon_vals = na_lon_vals
        bbox_extent = 'north america'

    if extent == 'Europe':
        lat_vals = eu_lat_vals
        lon_vals = eu_lon_vals
        bbox_extent = 'europe'

    if model_target == 'mda8':
        if metric == 'rmse':
            vmin = 0
            vmax= 20
        elif metric == 'residual':
            vmin = -25
            vmax = 25
        else:
            vmin = 10
            vmax = 90
    if model_target == 'bias':
        if metric == 'rmse':
            vmin = 0
            vmax= 20
        elif metric == 'residual':
            vmin = -25
            vmax = 25
        else:
            vmin = -20
            vmax = 50

    x, y = np.meshgrid(lon_vals, lat_vals, indexing='xy')
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    plt.pcolor(x, y, avg_data, cmap='coolwarm', vmin=vmin, vmax=vmax)
    ax.coastlines()
    #ax.stock_img()         # Commenting out so we can see the colours easier for comparison
    ax.set_extent(bbox_dict['{}'.format(bbox_extent)], crs=ccrs.PlateCarree())
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    im_ratio = len(lat_vals) / len(lon_vals)
    cb = plt.colorbar(pad=0.1)
    plt.tight_layout()
    plt.title('{} {} for {}, {} 2016 Test set'.format(model_target, metric, extent, anaylsis_date))
    # plt.show()
    plt.savefig('{}/{}_average_{}.png'.format(save_directory, model_target, metric))
    plt.close()

def generate_array_from_df(avg_list, variable, latitudes, longitudes, geo_extent, pred_target):
    """
    Generate array from df for spatial plotting
    avg_list: list of average values per lat, lon for the variable of interest
    variable: variable we are interested in plotting
    latitudes: list of latitude values
    longitudes: list of longitude values
    geo_extent: full name of geo extent to get correct lat, lon array size
    """
    na_lon_vals = [-124.875, -123.75, -122.625, -121.5, -120.375, -119.25,
                   -118.125, -117., -115.875, -114.75, -113.625, -112.5,
                   -111.375, -110.25, -109.125, -108., -106.875, -105.75,
                   -104.625, -103.5, -102.375, -101.25, -100.125, -99.,
                   -97.875, -96.75, -95.625, -94.5, -93.375, -92.25,
                   -91.125, -90., -88.875, -87.75, -86.625, -85.5,
                   -84.375, -83.25, -82.125, -81., -79.875, -78.75,
                   -77.625, -76.5, -75.375, -74.25, -73.125, -72.,
                   -70.875]

    na_lat_vals = [20.748, 21.869, 22.991, 24.112, 25.234, 26.355, 27.476, 28.598,
                   29.719, 30.841, 31.962, 33.084, 34.205, 35.327, 36.448, 37.57,
                   38.691, 39.813, 40.934, 42.056, 43.177, 44.299, 45.42, 46.542,
                   47.663, 48.785, 49.906, 51.028, 52.149, 53.271, 54.392]

    eu_lat_vals = [35.327, 36.448, 37.57, 38.691, 39.813, 40.934, 42.056, 43.177,
                   44.299, 45.42, 46.542, 47.663, 48.785, 49.906, 51.028, 52.149,
                   53.271, 54.392, 55.514, 56.635, 57.757, 58.878, 60., 61.121,
                   62.242, 63.364, 64.485]
    eu_lon_vals = [-9., -7.875, -6.75, -5.625, -4.5, -3.375, -2.25, -1.125,
                   0., 1.125, 2.25, 3.375, 4.5, 5.625, 6.75, 7.875,
                   9., 10.125, 11.25, 12.375, 13.5, 14.625, 15.75, 16.875,
                   18., 19.125, 20.25, 21.375, 22.5, 23.625, 24.75]


    df = pd.DataFrame()
    df['{}'.format(variable)] = avg_list
    df['lat'] = latitudes
    df['lon'] = longitudes

    if geo_extent == 'Europe':
        total_lon = 31
    if geo_extent == 'NorthAmerica':
        total_lon = 49

    var_list = []
    arr = None
    if pred_target == 'mda8':
        for i in range(0, len(df), total_lon):
            var_list_row = list(df.loc[i:i + total_lon-1]['{}'.format(variable)])
            var_list.append(var_list_row)
        arr = np.array(var_list)

    if pred_target == 'bias':
        # Need to account for the NaNs but have the output the size to match NA, EU extent
        if geo_extent == 'NorthAmerica':
            lons = na_lon_vals
            lats = na_lat_vals
            num_lats = 31
            num_lons = 49
            total_elements = num_lons * num_lats
        if geo_extent == 'Europe':
            lons = eu_lon_vals
            lats = eu_lat_vals
            num_lats = 27
            num_lons = 31
            total_elements = num_lons * num_lats

        # create dummy xarray of nans
        x = xr.DataArray(np.arange(total_elements).reshape(num_lats, num_lons),
                         dims=["lat", "lon"], coords={'lat': lats, 'lon': lons})
        ds_zeros = xr.zeros_like(x)
        ds_nans = ds_zeros.where(ds_zeros != 0)
        # Now let's update for every point the data array that is non NaN
        for i in range(len(df)):
            lon_point = df.loc[i]['lon']
            lat_point = df.loc[i]['lat']
            lon_idx = np.where(ds_zeros['lon'].values == lon_point)
            lat_idx = np.where(ds_zeros['lat'].values == lat_point)
            ds_nans[{'lon': lon_idx[0][0], 'lat': lat_idx[0][0]}] = df.loc[i]['{}'.format(variable)]

        arr = ds_nans.to_numpy()

    return arr


def calculating_spatial_results(df_sorted, target, analysis_date, save_directory, geo_region):
    """
    Plotting spatial rmse
    df_sorted: sorted dataframe saved during run for lat, lon points
    analysis_date: will factor in for number of samples, 31 for july, august, 30 for june, 92 for summer
    """
    date_dict = {
        'june': 30,
        'july': 31,
        'august': 31,
        'summer': 92
    }
    num_samples = int(date_dict['{}'.format(analysis_date)])

    # Take the average over all samples for the same point
    avg_rmse_list = []
    avg_pred_list = []
    avg_gt_list = []
    lat_points = []
    lon_points = []


    if target == 'mda8':
        for i in range(0, len(df_sorted), num_samples):
            # calc average over the lat, lon point
            avg_rmse = np.mean(df_sorted['rmse'][i:i+num_samples])
            avg_pred = np.mean(df_sorted['pred'][i:i+num_samples])
            avg_label = np.mean(df_sorted['label'][i:i+num_samples])

            # append to list
            avg_rmse_list.append(avg_rmse)
            avg_pred_list.append(avg_pred)
            avg_gt_list.append(avg_label)
            lat_points.append(df_sorted.loc[i]['lat'])
            lon_points.append(df_sorted.loc[i]['lon'])

    if target == 'bias':
        # We will have nans removed so need to be more careful in calculating avg per lat, lon point
        unique_points = []
        for i in range(len(df_sorted)):
            point = (df_sorted.loc[i]['lat'], df_sorted.loc[i]['lon'])
            # Check if point in list already
            if point in unique_points:
                continue
            else:
                # Append unique point to list
                unique_points.append(point)
                # Take average over subset
                df_subset = df_sorted.loc[(df_sorted['lat'] == point[0]) & (df_sorted['lon'] == point[1])]
                avg_rmse = np.mean(df_subset['rmse'])
                avg_pred = np.mean(df_subset['pred'])
                avg_label = np.mean(df_subset['label'])

                avg_rmse_list.append(avg_rmse)
                avg_pred_list.append(avg_pred)
                avg_gt_list.append(avg_label)
                lat_points.append(point[0])
                lon_points.append(point[1])

    # Get average lists as arrays
    rmse_arr = generate_array_from_df(avg_rmse_list, 'avg_rmse', lat_points, lon_points, geo_region, target)
    pred_arr = generate_array_from_df(avg_pred_list, 'avg_pred', lat_points, lon_points, geo_region, target)
    gt_arr = generate_array_from_df(avg_gt_list, 'avg_gt', lat_points, lon_points, geo_region, target)
    residual_arr = pred_arr - gt_arr

    rmse_arr = rmse_arr.astype('float32')
    pred_arr = pred_arr.astype('float32')
    gt_arr = gt_arr.astype('float32')
    residual_arr = residual_arr.astype('float32')

    np.save('{}/groundtruth_avg.npy'.format(save_directory), gt_arr)

    # Now we can plot, so let's call this function for the predictions, groundtruth and rmse
    plotting_spatial_data(rmse_arr, 'rmse', target, analysis_date, save_directory, geo_region)
    plotting_spatial_data(pred_arr, 'prediction', target,  analysis_date, save_directory, geo_region)
    plotting_spatial_data(gt_arr, 'groundtruth', target, analysis_date, save_directory, geo_region)
    plotting_spatial_data(residual_arr, 'residual', target, analysis_date, save_directory, geo_region)


# Read in file of interest
aoi = 'NorthAmerica'
target = 'mda8'
num_channels = 9
analysis_period = 'summer'
'''
results_dir = '/Users/kelseyd/Desktop/random_forest/runs/{}/{}/{}channels/{}'.format(aoi, target, num_channels,
                                                                                analysis_period)
'''
results_dir = '/Users/kelseydoerksen/Desktop/sudsaq/rf/runs/{}/mda8/{}channels/summer'.format(aoi, num_channels)


# Subset yhat (predictions), y_true (groundtruth)
results_df = pd.read_csv('{}/prediction_groundtruth_rmse.csv'.format(results_dir))

# Change to type np.float.32 to match UNet
results_df['label'] = results_df['label'].astype('float32')
results_df['pred'] = results_df['pred'].astype('float32')

yhat = results_df['pred']
y_true = results_df['label']

# --- Plot results ---

# Spatial Map

print('plotting spatial results')
#calculating_spatial_results(results_df, target, analysis_period, results_dir, aoi)

# Plot true vs predicted
#print('Plotting y_test vs y_hat')
#truth_vs_predicted(y_true, yhat, results_dir, aoi)


# Plot histogram
#print('plotting histogram')
plot_histogram(y_true, yhat, results_dir, aoi, analysis_period)

# Plot importances -> To update
#plot_importances(importances, perm_importances, results_dir, analysis_period, 2016, aoi)