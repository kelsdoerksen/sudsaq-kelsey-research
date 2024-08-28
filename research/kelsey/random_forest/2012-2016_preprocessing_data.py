"""
Preprocessing data to run in random forest that spans
2012-2016 to include nightlight in the feature space as
well as latitude, longitude point
"""

import pandas as pd

data_dir = '/Volumes/PRO-G40/sudsaq/random_forest/rf_sample_data_northamerica/gee'
save_dir = '/Volumes/PRO-G40/sudsaq/random_forest/rf_sample_data_northamerica/gee_only_combined'

years = ['2012', '2013', '2014', '2015', '2016']
months = ['june', 'july', 'august']

def combine_gee_features(year, month, data_directory):
    """
    combine all gee features into one df
    """
    modis = pd.read_csv('{}/{}_{}_modis_features.csv'.format(data_directory, year, month))
    modis = modis.drop(columns='Unnamed: 0')
    pop = pd.read_csv('{}/{}_{}_population_features.csv'.format(data_directory, year, month))
    pop = pop.drop(columns='Unnamed: 0')
    nightlight_rad = pd.read_csv('{}/{}_{}_avg_rad_nightlight_features.csv'.format(data_directory, year, month))
    nightlight_rad = nightlight_rad.drop(columns='Unnamed: 0')
    nightlight_cvg = pd.read_csv('{}/{}_{}_cf_cvg_nightlight_features.csv'.format(data_directory, year, month))
    nightlight_cvg = nightlight_cvg.drop(columns='Unnamed: 0')

    features_combined = pd.concat([modis, pop, nightlight_rad, nightlight_cvg], axis=1)
    return features_combined


for y in years:
    for m in months:
        combined_features = combine_gee_features(y, m, data_dir)
        combined_features.to_csv('{}/{}_{}_combined_gee.csv'.format(save_dir, y, m))