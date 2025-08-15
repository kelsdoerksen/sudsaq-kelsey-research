"""
@author: kdoerksen
Runs simple RF model for comparison with UNet model
Good for quick experiments.
Hard coding things for now because I want to use this
as a direct comparison to the UNet model, training years 2005-2015,
testing year 2016
"""

import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
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
import pickle
import joblib
import wandb

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
    parser = argparse.ArgumentParser(description='Running RF for bias or mda8')
    parser.add_argument('--root_dir', help='Root directory of data')
    parser.add_argument('--month', help='Time of year to run analysis, must be one of '
                                                'june, july, august', required=True)
    parser.add_argument('--target', help='Target to predict, must be one of: mda8, bias', required=True)
    parser.add_argument('--region', help='Geographic region, currently supports north_america or europe', required=True)
    parser.add_argument('--save_dir', help='Save Directory for run', required=True)
    parser.add_argument('--test_year', help='Test year', required=True)
    parser.add_argument('--n_channels', help='Number of channels', required=True)
    parser.add_argument('--tuning', help='Specify if tuning model', default=None)
    return parser.parse_args()

def calculate_rmse(preds, label, lat_list, lon_list, analysis_date, save_directory, wandb_experiment):
    """
    Calculate rmse for preds, groundtruth and save for future plotting
    preds: predictions from rf model
    lat_list: list of corresponding latitude values to preserve spatial information
    lon_list: list of corresponding longitude values to preserve spatial information
    analysis_date: will factor in for number of samples, 31 for july, august, 30 for june, 92 for summer
    save_directory: where to save
    """
    date_dict = {
        'June': 30,
        'July': 31,
        'Aug': 31,
        'Summer': 92
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

    df_sorted.to_csv('{}/{}_prediction_groundtruth_rmse.csv'.format(save_directory, wandb_experiment.name), index=False)


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

def run_rf(X_train, y_train, X_val, y_val, X_test, save_dir, tuning=False):
    """
    Run rf
    """
    if not tuning:
        model_state = 'No tuning'
        # Create an instance of Random Forest
        rf = RandomForestRegressor(n_estimators=100,
                                   max_features=int(0.3 * (len(var_names))),
                                   random_state=300,
                                   verbose=1)

        out_model = os.path.join(save_dir, 'random_forest_predictor.joblib')
        dump(rf, out_model)
        print('No model hyperparameter tuning')
        # Concat the train and validation sets together to train on the entire available dataset
        X_train = pd.concat([X_train, X_val], axis=0)
        y_train = pd.concat([y_train, y_val], axis=0)

        # Fit the model
        print('Fitting model...')
        rf.fit(X_train, y_train)
    else:
        # Tune the model
        model_state = 'tuned_best_params'
        print('Tuning the model')
        param_grid = {
            'bootstrap': [True],
            'max_depth': [3, 5, 7],
            'min_samples_split': [4, 6, 8],
            'n_estimators': [100, 200, 300]
        }

        tune_list = []
        for i in range(3):
            print('Tuning for iteration: {}'.format(i))
            # Create an instance of Random Forest
            rf = RandomForestRegressor(n_estimators=100,
                                       max_features=int(0.3 * (len(var_names))),
                                       random_state=300,
                                       verbose=1)
            # grid search cv
            rf_cv = GridSearchCV(estimator=rf,
                                 param_grid=param_grid,
                                 scoring=['accuracy'],
                                 refit='accuracy',
                                 cv=3,
                                 n_jobs=-1)

            # Fit the grid search to the data
            print('Running grid search cv on training set...')
            rf_cv.fit(X_train, y_train)

            # Set model to the best estimator from grid search
            best_forest = rf_cv.best_estimator_

            # Prediction on validation set with best forest
            tuned_probs = best_forest.predict_proba(X_val)
            accuracy = best_forest.score(X_val, y_val)

            forest_dict = rf_cv.best_params_
            forest_dict['accuracy'] = accuracy
            forest_dict['best_model'] = best_forest
            tune_list.append(forest_dict)

        # Get max F1 and use this as our best model
        max_dict = max(tune_list, key=lambda x: x['f1'])
        print('Max model stats after parameter tuning is: {}'.format(max_dict))
        # Save the model to file
        joblib.dump(max_dict['best_model'], '{}/rf_model.joblib'.format(save_dir))
        # saving as pickle too
        with open("{}/rf_model.pkl".format(save_dir), "wb") as file:
            pickle.dump(max_dict['best_model'], file)
        forest = max_dict['best_model']

        # Fit the model
        print('Fitting best model...')
        forest.fit(X_train, y_train)


    yhat = rf.predict(X_test)
    return yhat, rf


def load_summer(root_dir, region):
    """
    Loading summer data
    """
    # --- Combine the training data
    # Load the training samples
    june_sample_train = pd.read_csv('{}/samples/2005-2015_train_{}_June_momo_gee_features.csv'.format(root_dir, region))
    july_sample_train = pd.read_csv('{}/samples/2005-2015_train_{}_July_momo_gee_features.csv'.format(root_dir, region))
    aug_sample_train = pd.read_csv('{}/samples/2005-2015_train_{}_Aug_momo_gee_features.csv'.format(root_dir, region))

    # Load the training labels
    june_label_train = pd.read_csv('{}/target/{}/{}/June_2005-2015_target_with_coords.csv'.format(root_dir, region, target))
    july_label_train = pd.read_csv(
        '{}/target/{}/{}/July_2005-2015_target_with_coords.csv'.format(root_dir, region, target))
    aug_label_train = pd.read_csv(
        '{}/target/{}/{}/Aug_2005-2015_target_with_coords.csv'.format(root_dir, region, target))

    # Concatenate together
    train_data = pd.concat([june_sample_train, july_sample_train, aug_sample_train], axis=0)
    train_label = pd.concat([june_label_train, july_label_train, aug_label_train], axis=0)

    train_combined = pd.concat([train_data, train_label], axis=1)
    train_combined = train_combined.drop(columns=['Unnamed: 0.1', 'Unnamed: 0'])
    train_combined = train_combined.dropna()
    train_features = train_combined.drop(columns=['lat', 'lon', 'target'])
    train_labels = train_combined['target']

    # --- Combine the testing data
    june_sample_test = pd.read_csv('{}/samples/2016_test_{}_June_momo_gee_features.csv'.format(root_dir, region))
    july_sample_test = pd.read_csv('{}/samples/2016_test_{}_July_momo_gee_features.csv'.format(root_dir, region))
    aug_sample_test = pd.read_csv('{}/samples/2016_test_{}_Aug_momo_gee_features.csv'.format(root_dir, region))

    # Load the training labels
    june_label_test = pd.read_csv(
        '{}/target/{}/{}/June_2016_target_with_coords.csv'.format(root_dir, region, target))
    july_label_test = pd.read_csv(
        '{}/target/{}/{}/July_2016_target_with_coords.csv'.format(root_dir, region, target))
    aug_label_test = pd.read_csv(
        '{}/target/{}/{}/Aug_2016_target_with_coords.csv'.format(root_dir, region, target))

    # Concatenate together
    test_data = pd.concat([june_sample_test, july_sample_test, aug_sample_test], axis=0)
    test_label = pd.concat([june_label_test, july_label_test, aug_label_test], axis=0)

    test_combined = pd.concat([test_data, test_label], axis=1)
    test_combined = test_combined.drop(columns=['Unnamed: 0.1', 'Unnamed: 0'])
    test_combined = test_combined.dropna()
    lat_list = list(test_combined['lat'].iloc[:, 0])
    lon_list = list(test_combined['lon'].iloc[:, 0])
    test_features = test_combined.drop(columns=['lat', 'lon', 'target'])
    test_labels = test_combined['target']

    return train_features, train_labels, test_features, test_labels, lat_list, lon_list

if __name__ == '__main__':
    args = get_args()
    month = args.month
    target = args.target
    region = args.region
    save_dir = args.save_dir
    root_dir = args.root_dir
    test_year = args.test_year
    n_channels = int(args.n_channels)
    tuning = args.tuning

    # Setting up wandb - project is U-Net Test but whatever I already have this setup
    experiment = wandb.init(project='U-Net Test', resume='allow', anonymous='must')
    experiment.config.update(
        dict(analysis_month=month, target=target, region=region, model='rf', test_year=test_year,
             n_channels=n_channels))

    save_dir = '{}/{}'.format(save_dir, experiment.name)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    if month == 'Summer':
        train_features, train_labels, X_test, y_test, lat_list, lon_list = load_summer(root_dir, region)
    else:
        # --- Loading Training Data ---
        train_data = pd.read_csv('{}/samples/2005-2015_train_{}_{}_momo_gee_features.csv'.format(root_dir, region, month))
        train_label = pd.read_csv('{}/target/{}/{}/{}_2005-2015_target_with_coords.csv'.format(root_dir, region, target, month))
        train_combined = pd.concat([train_data, train_label], axis=1)
        train_combined = train_combined.drop(columns=['Unnamed: 0.1', 'Unnamed: 0'])
        train_combined = train_combined.dropna()
        train_features = train_combined.drop(columns=['lat', 'lon', 'target'])
        train_labels = train_combined['target']

        # --- Loading Testing Data ---
        test_data = pd.read_csv('{}/samples/2016_test_{}_{}_momo_gee_features.csv'.format(root_dir, region, month))
        test_label = pd.read_csv('{}/target/{}/{}/{}_2016_target_with_coords.csv'.format(root_dir, region, target, month))
        test_combined = pd.concat([test_data, test_label], axis=1)
        test_combined = test_combined.drop(columns=['Unnamed: 0.1', 'Unnamed: 0'])
        test_combined = test_combined.dropna()
        lat_list = list(test_combined['lat'].iloc[:,0])
        lon_list = list(test_combined['lon'].iloc[:,0])
        X_test = test_combined.drop(columns=['lat', 'lon', 'target'])
        y_test = test_combined['target']

    X_train, X_val, y_train, y_val = train_test_split(train_features, train_labels, test_size=0.10,
                                                      random_state=np.random.RandomState())

    print('Number of train samples: {}'.format(len(X_train)))
    print('Number of test samples: {}'.format(len(X_test)))

    if n_channels == 28:
        print('Running experiment with only momo features...')
        columns_to_drop = X_train.columns[~X_train.columns.str.contains('momo', case=False)]
        X_train = X_train.drop(columns=columns_to_drop)
        X_val = X_val.drop(columns=columns_to_drop)
        X_test = X_test.drop(columns=columns_to_drop)

    var_names = X_train.columns.values.tolist()

    yhat, rf = run_rf(X_train, y_train, X_val, y_val, X_test, save_dir, tuning=tuning)
    calculate_rmse(yhat, y_test, lat_list, lon_list, month, save_dir, experiment)

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

    with open('{}/{}_results.txt'.format(save_dir, experiment.name), 'w') as f:
        f.write(' rmse is {}'.format(rmse))
        f.write(' mean absolute percentage error is: {}'.format(mape))
        #f.write(" r correlation is: {}".format(r))
        #f.write(' r2 score is: {}'.format(r2))

    # Calculate importances and save
    importances = calc_importances(rf, var_names)
    importances.to_csv('{}/{}_feature_importances.csv'.format(save_dir, experiment.name))


    # Calculate permutation importances and save -> To update
    #perm_importances = calc_perm_importance(rf, X_test, y_test, var_names, results_dir)
    #perm_importances.to_csv('{}/feature_importances_perm.csv'.format(results_dir))
