"""
@author: kdoerksen
Runs simple RF model for comparison with UNet model
Good for quick experiments.
Hard coding things for now because I want to use this
as a direct comparison to the UNet model, training years 2005-2015,
testing year 2016
"""

import numpy as np
from sklearn.utils import shuffle
import os
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.metrics    import (
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score
)
from joblib import dump
from sklearn.inspection import permutation_importance
import argparse
import math

root_dir = '/Users/kelseydoerksen/Desktop/sudsaq_rf'

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

def get_args():
    parser = argparse.ArgumentParser(description='Running UNet for mda8 or bias target')
    parser.add_argument('--analysis_time', help='Time of year to run analysis, must be one of '
                                                'june, july, august, or summer', required=True)
    parser.add_argument('--target', help='Target to predict, must be one of: mda8, bias', required=True)
    parser.add_argument('--n_features', help='Number of features, currently supports 9, 32, 39', required=True)
    parser.add_argument('--region', help='Geographic region, currently supports na or eu', required=True)

    return parser.parse_args()


def calculate_rmse(preds, label, lat_list, lon_list, analysis_date, save_directory):
    """
    Calculate rmse for preds, groundtruth and save for future plotting
    preds: predictions from rf model
    lat_list: list of corresponding latitude values to preserve spatial information
    lon_list: list of corresponding longitude values to preserve spatial information
    analysis_date: will factor in for number of samples, 31 for july, august, 30 for june, 92 for summer
    save_directory: where to save
    """
    date_dict = {
        'june': 30,
        'july': 31,
        'august': 31,
        'summer': 92
    }
    num_samples = int(date_dict['{}'.format(analysis_date)])

    df = pd.DataFrame()
    df['pred'] = preds
    df['label'] = label.values
    df['lat'] = lat_list
    df['lon'] = lon_list
    df_sorted = df.sort_values(by=['lat', 'lon'])
    df_sorted = df_sorted.reset_index()

    rmse_list = []
    for i in range(len(df_sorted)):
        if np.isnan(df_sorted.loc[i]['label']):
            rmse_list.append(np.nan)
        else:
            mse = np.square(np.subtract(df_sorted.loc[i]['label'], df_sorted.loc[i]['pred'])).mean()
            rmse = math.sqrt(mse)
            rmse_list.append(rmse)

    df_sorted['rmse'] = rmse_list

    df_sorted.to_csv('{}/prediction_groundtruth_rmse.csv'.format(save_directory))


def calc_perm_importance(model, data, target, feature_names, dir):
    permimp = permutation_importance(model, data, target, random_state=0)
    # Only want the summaries, remove the importances array
    del permimp['importances']

    # Convert to a DataFrame and sort by importance value
    df = pd.DataFrame(permimp.values(), columns=feature_names, index=['importance', 'stddev'])
    df = df.sort_values(by='importance', axis=1, ascending=False)
    fmt = {}
    for var, vals in df.items():
        fmt[var] = f'{vals.importance} +/- {vals.stddev}'

    '''
    Logger.info('Permutation importance +/- stddev:')
    strings = align_print(fmt, enum=True, print=Logger.info)
    with open('{}/perm_importances.txt'.format(dir), 'w') as file:
        file.write('\n\nPermutation Feature Importance:\n')
        file.write('\n'.join(strings))
    '''

    return df


def get_X_and_y_train(df_feature_list, df_target_list):
    """
    Given list of feature, target dataframes, return
    X dataframe of features
    y dataframe of target
    """
    df_data = pd.concat(df_feature_list)
    df_target = pd.concat(df_target_list)
    df_data = df_data.drop(columns=['Unnamed: 0'])
    df_target = df_target.drop(columns=['Unnamed: 0'])

    df = df_data
    df['target'] = df_target['target']

    # Remove NaN rows if they exist in target
    df = df.dropna(axis=0)

    df = shuffle(df)

    y = df['target']
    X = df.drop(columns=['target'])

    return X, y

def get_X_and_y_test(df_feature_list, df_target_list):
    """
    Given list of feature, target dataframes, return
    X dataframe of features
    y dataframe of target
    """
    df_data = pd.concat(df_feature_list)
    df_target = pd.concat(df_target_list)
    df_data = df_data.drop(columns=['Unnamed: 0'])
    df_target = df_target.drop(columns=['Unnamed: 0'])

    df = df_data
    df['target'] = df_target['target']
    df['lat'] = df_target['lat']
    df['lon'] = df_target['lon']

    # Remove NaN rows if they exist in target
    df = df.dropna(axis=0)

    df = shuffle(df)
    y = df[['target', 'lat', 'lon']]
    X = df.drop(columns=['target', 'lat', 'lon'])

    return X, y

def load_train_data(analysis_time, pred_target, n_features, region):
    """
    Loading training data accordingly to analysis time set
    Takes in analysis time (june, jul, aug, summer)
    pred_target (bias or mda8),
    n_features (9, 32, 39 supported)
    Hard coded that 2005-2015 is the training data
    """

    sample_dir = '/Volumes/PRO-G40/sudsaq/random_forest/rf_sample_data_northamerica/gee_only_combined'
    target_dir = '/Volumes/PRO-G40/sudsaq/random_forest/tmp_target'

    sample_list = []
    years = ['2012', '2013', '2014', '2015']
    for y in years:
        sample_list.append(pd.read_csv('{}/{}_august_combined_gee.csv'.format(sample_dir, y)))

    df_samples = pd.concat(sample_list)
    df_labels = pd.read_csv('{}/august_2012-2015_target.csv'.format(target_dir))


    '''
    df_samples = []
    df_labels = []

    sample_dir = '{}/data/samples/{}/{}features'.format(root_dir, region, n_features)
    target_dir = '{}/data/target/{}/{}'.format(root_dir, region, pred_target)

    samples_june = pd.read_csv('{}/june_2005-2015_features.csv'.format(sample_dir))
    samples_july = pd.read_csv('{}/july_2005-2015_features.csv'.format(sample_dir))
    samples_august = pd.read_csv('{}/august_2005-2015_features.csv'.format(sample_dir))

    target_june = pd.read_csv('{}/june_2005-2015_target.csv'.format(target_dir))
    target_july = pd.read_csv('{}/july_2005-2015_target.csv'.format(target_dir))
    target_august = pd.read_csv('{}/august_2005-2015_target.csv'.format(target_dir))
    
    


    if analysis_time == 'summer':
        df_samples = [samples_june, samples_july, samples_august]
        df_labels = [target_june, target_july, target_august]
    if analysis_time == 'june':
        df_samples = [samples_june]
        df_labels = [target_june]
    if analysis_time == 'july':
        df_samples = [samples_july]
        df_labels = [target_july]
    if analysis_time == 'august':
        df_samples = [samples_august]
        df_labels = [target_august]
    '''

    return df_samples, df_labels


def calc_importances(model, feature_names):
    '''
    Calculate feature importances, save as txt
    and plot
    '''
    importances = model.feature_importances_
    stddev = np.std([est.feature_importances_ for est in model.estimators_], axis=0)
    df = pd.DataFrame(np.array([importances, stddev]), columns=feature_names, index=['importance', 'stddev'])
    df = df.sort_values(by='importance', axis=1, ascending=False)
    '''
    fmt = {}
    for var, vals in df.items():
        fmt[var] = f'{vals.importance} +/- {vals.stddev}'
    Logger.info('Permutation importance +/- stddev:')
    strings = align_print(fmt, enum=True, print=Logger.info)
    with open('{}/importances.txt'.format(dir), 'w') as file:
        file.write('\n\nFeature Importance:\n')
        file.write('\n'.join(strings))
    '''

    return df


def load_test_data(analysis_time, pred_target, n_features, region):
    """
    Loading testing data accordingly to analysis time set
    Takes in analysis time (june, jul, aug, summer)
    pred_target (bias or mda8),
    n_features (9, 32, 39 supported)
    Hard coded that 2016 is test year
    """

    test_labels = pd.read_csv('/Volumes/PRO-G40/sudsaq/random_forest/target/NorthAmerica/mda8/august_2016_target_with_coords.csv')
    test_samples = pd.read_csv('/Volumes/PRO-G40/sudsaq/random_forest/rf_sample_data_northamerica/gee_only_combined/'
                               '2016_august_combined_gee.csv')

    '''
    df_samples = []
    df_labels = []

    sample_dir = '{}/data/samples/{}/{}features'.format(root_dir, region, n_features)
    target_dir = '{}/data/target/{}/{}'.format(root_dir, region, pred_target)

    samples_june = pd.read_csv('{}/june_2016_features.csv'.format(sample_dir))
    samples_july = pd.read_csv('{}/july_2016_features.csv'.format(sample_dir))
    samples_august = pd.read_csv('{}/august_2016_features.csv'.format(sample_dir))

    target_june = pd.read_csv('{}/june_2016_target_with_coords.csv'.format(target_dir))
    target_july = pd.read_csv('{}/july_2016_target_with_coords.csv'.format(target_dir))
    target_august = pd.read_csv('{}/august_2016_target_with_coords.csv'.format(target_dir))

    if analysis_time == 'summer':
        df_samples = [samples_june, samples_july, samples_august]
        df_labels = [target_june, target_july, target_august]
    if analysis_time == 'june':
        df_samples = [samples_june]
        df_labels = [target_june]
    if analysis_time == 'july':
        df_samples = [samples_july]
        df_labels = [target_july]
    if analysis_time == 'august':
        df_samples = [samples_august]
        df_labels = [target_august]

    return df_samples, df_labels
    '''
    return test_samples, test_labels

if __name__ == '__main__':
    args = get_args()
    analysis_period = args.analysis_time
    target = args.target
    aoi = args.region
    num_features = args.n_features
    results_dir = args.results_dir

    if aoi == 'na':
        full_geo_name = 'NorthAmerica'
    if aoi == 'eu':
        full_geo_name = 'Europe'
    if aoi == 'globe':
        full_geo_name = 'Globe'

    # --- Loading Training Data ---
    train_df_data, train_df_target = load_train_data(analysis_period, target, num_features, full_geo_name)
    #X_train, y_train = get_X_and_y_train(train_df_data, train_df_target)

    # --- Loading Testing Data ---
    test_df_data, test_df_target = load_test_data(analysis_period, target, num_features, full_geo_name)
    #X_test, y_test = get_X_and_y_test(test_df_data, test_df_target)

    #lat_list = list(y_test['lat'])
    #lon_list = list(y_test['lon'])

    if num_features == '39':
        train_cols_to_drop = ['index', 'index.1']
        X_train = X_train.drop(columns=train_cols_to_drop)
    if num_features == '32':
        train_cols_to_drop = ['Unnamed: 0.1', 'index', 'index.1']
        X_train = X_train.drop(columns=train_cols_to_drop)
        X_test = X_test.drop(columns=['Unnamed: 0.1'])
    if num_features == '9':
        train_cols_to_drop = ['Unnamed: 0.1']
        X_train = X_train.drop(columns=train_cols_to_drop)
        X_test = X_test.drop(columns=['Unnamed: 0.1'])

    y_test = y_test.drop(columns=['lat', 'lon'])

    var_names = X_train.columns.values.tolist()

    print('Number of train samples: {}'.format(len(X_train)))
    print('Number of test samples: {}'.format(len(X_test)))


    rf = RandomForestRegressor(n_estimators=100,
                                   max_features=int(0.3*(len(var_names))),
                                   random_state=300,
                                   verbose=1)

    rf.fit(X_train, y_train)
    out_model = os.path.join(results_dir, 'random_forest_predictor.joblib')
    dump(rf, out_model)

    yhat = rf.predict(X_test)

    calculate_rmse(yhat, y_test, lat_list, lon_list, analysis_period, results_dir)

    # Calculate rmse for entire run
    mse = mean_squared_error(y_test, yhat)
    rmse = math.sqrt(mse)
    print('rmse is {}'.format(rmse))

    # Calculate mape
    mape = mean_absolute_percentage_error(y_test, yhat)
    print('mean absolute percentage error is: {}'.format(mape))

    # Calculate r correlation value
    #r = pearsonr(y_test, yhat)[0]
    #print("r correlation is: {}".format(r))

    # Calculate r2 score
    #r2 = r2_score(y_test,yhat)
    #print('r2 score is: {}'.format(r2))

    with open('{}/results.txt'.format(results_dir), 'w') as f:
        f.write(' rmse is {}'.format(rmse))
        f.write(' mean absolute percentage error is: {}'.format(mape))
        #f.write(" r correlation is: {}".format(r))
        #f.write(' r2 score is: {}'.format(r2))

    # Calculate importances and save
    importances = calc_importances(rf, var_names)
    importances.to_csv('{}/feature_importances.csv'.format(results_dir))


    # Calculate permutation importances and save -> To update
    #perm_importances = calc_perm_importance(rf, X_test, y_test, var_names, results_dir)
    #perm_importances.to_csv('{}/feature_importances_perm.csv'.format(results_dir))

    '''
    # OLD
    if param_tuning:
        Logger.info('Finding optimal hyperparameters')
        params = [{
                    'n_estimators': [50, 100, 150, 200],
                    'max_depth': [3, 4, 5]
                }]
        kfold = KFold(n_splits=5, shuffle=True, random_state=1234)
        clf = GridSearchCV(
            RandomForestRegressor(), params, cv=kfold,
            scoring='neg_root_mean_squared_error', n_jobs=10, error_score='raise'
        )
        clf.fit(X, y)
        print('Parameter selection:')
        print(f'n_estimators: {clf.best_params_["n_estimators"]}')
        print(f'max_depth: {clf.best_params_["max_depth"]}')
    
        # Create the Gradient Boosting predictor
        Logger.info('Training Random Forest')
        rf = RandomForestRegressor(n_estimators=clf.best_params_['n_estimators'],
                                   max_depth=clf.best_params_['max_depth'],
                                   random_state=300,
                                   verbose=1)
    else:
        rf = RandomForestRegressor(n_estimators=100,
                                   max_features=int(0.3*(len(var_names))),
                                   random_state=300,
                                   verbose=1)
    
    '''