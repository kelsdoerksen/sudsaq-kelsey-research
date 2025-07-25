"""
Scratch script to combine momo and gee data
(eventually probably merge this into one preprocess script, but fine for now)
"""

import numpy as np
import pandas as pd
import os

def get_gee_train_data(gee_data, month, gee_directory, years):
    """
    Get gee train data (2005-2015)
    """
    df_list = []
    for y in years:
        df_list.append(pd.read_csv('{}/{}_{}_{}_features.csv'.format(gee_directory, y, month, gee_data)))

    df_features = pd.concat(df_list)
    return df_features


def combine_train_data(query_month, geo_extent, num_momo_channels, training_years):

    if geo_extent == 'na':
        full_geo = 'NorthAmerica'
    if geo_extent == 'eu':
        full_geo = 'Europe'

    total_channels = 23 + num_momo_channels

    # Set directories
    gee_dir = '/Users/kelseyd/Desktop/random_forest/data/samples/{}/gee'.format(full_geo)
    momo_dir = '/Users/kelseyd/Desktop/random_forest/data/samples/{}/{}features'.format(full_geo,
                                                                                        str(num_momo_channels))
    save_dir = '/Users/kelseyd/Desktop/random_forest/data/samples/{}/{}features'.format(full_geo, str(total_channels))

    # Load and combine data
    momo_train_data = pd.read_csv('{}/{}_2005-2015_features.csv'.format(momo_dir, query_month))
    modis_train_data = get_gee_train_data('modis', query_month, gee_dir, training_years)
    pop_train_data = get_gee_train_data('population', query_month, gee_dir, training_years)
    momo_train_data = momo_train_data.drop(columns=['Unnamed: 0'])
    modis_train_data = modis_train_data.drop(columns=['Unnamed: 0']).reset_index()
    pop_train_data = pop_train_data.drop(columns=['Unnamed: 0']).reset_index()
    combined = pd.concat([momo_train_data, modis_train_data, pop_train_data], axis=1)
    combined.to_csv('{}/{}_momo_gee_2005-2015_features.csv'.format(save_dir, query_month))


def combine_test_data(test_query_month, geo_extent, num_momo_channels, test_query_year):

    if geo_extent == 'na':
        full_geo = 'NorthAmerica'
    if geo_extent == 'eu':
        full_geo = 'Europe'

    total_channels = 23 + num_momo_channels

    # Set directories
    gee_dir = '/Users/kelseyd/Desktop/random_forest/data/samples/{}/gee'.format(full_geo)
    momo_dir = '/Users/kelseyd/Desktop/random_forest/data/samples/{}/{}features'.format(full_geo,
                                                                                        str(num_momo_channels))
    save_dir = '/Users/kelseyd/Desktop/random_forest/data/samples/{}/{}features'.format(full_geo, str(total_channels))

    # Load and combine data
    momo_test_data = pd.read_csv('{}/{}_{}_features.csv'.format(momo_dir, test_query_month, test_query_year))
    modis_test_data = pd.read_csv('{}/{}_{}_modis_features.csv'.format(gee_dir, test_query_year, test_query_month))
    pop_test_data = pd.read_csv('{}/{}_{}_population_features.csv'.format(gee_dir, test_query_year, test_query_month))
    momo_test_data = momo_test_data.drop(columns=['Unnamed: 0'])
    modis_test_data = modis_test_data.drop(columns=['Unnamed: 0'])
    pop_test_data = pop_test_data.drop(columns=['Unnamed: 0'])
    combined_test = pd.concat([momo_test_data, modis_test_data, pop_test_data], axis=1)
    combined_test.to_csv('{}/{}_momo_gee_{}_features.csv'.format(save_dir, test_query_month, test_query_year))


months = ['june', 'july', 'august']
train_years = ['2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012',
         '2013', '2014', '2015']
test_year = '2016'

geo = 'eu'
momo_channels = [9, 16]
for m in months:
    for channel in momo_channels:
        combine_train_data(m, geo, channel, train_years)
        combine_test_data(m, geo, channel, test_year)








