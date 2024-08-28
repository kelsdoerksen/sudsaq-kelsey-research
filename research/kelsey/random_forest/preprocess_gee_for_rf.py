"""
Scratch script to preprocess GEE in RF format
"""
import xarray as xr
import numpy as np
import pandas as pd

combined_data_dir = '/Users/kelseyd/Desktop/random_forest/data/momo_and_gee'

years = ['2005','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015','2016']
months_dict = {'june': 30,
               'july': 31,
               'aug': 31}


def generate_features(x, gee_data, gee_band, month_query):
    """
    x: xarray dataset
    gee_data: gee data name
    gee_band: gee data band
    """
    features = []
    if gee_data == 'nightlight':
        features = ['nightlight.{}.var'.format(gee_band), 'nightlight.{}.mean'.format(gee_band),
                    'nightlight.{}.max'.format(gee_band), 'nightlight.{}.min'.format(gee_band)]
        data = x.to_array().stack({'loc': ['lat', 'lon']})
        data = data.transpose('loc', 'variable')
        df = pd.DataFrame(data=data.values, columns=features)
        month_length = months_dict[month_query]
        df_monthly = pd.concat([df] * month_length)
        return df_monthly

    dummy_date = x.time.values[0]  # Dummy date to filter since GEE modis, pop is constant vals
    subset_ds = x.sel(time=dummy_date)
    if gee_data == 'modis':
        features = ['modis.mode', 'modis.var', 'modis.evg_conif', 'modis.evg_broad', 'modis.dcd_needle',
                    'modis.dcd_broad', 'modis.mix_forest', 'modis.cls_shrub', 'modis.open_shrub', 'modis.woody_savanna',
                    'modis.savanna', 'modis.grassland', 'modis.perm_wetland', 'modis.cropland', 'modis.urban',
                    'modis.crop_nat_veg', 'modis.perm_snow', 'modis.barren', 'modis.water_bds']
    if gee_data == 'pop':
        features = ['pop.var', 'pop.mean', 'pop.max', 'pop.min']

    data = subset_ds.to_array().stack({'loc': ['lat', 'lon']})
    data = data.transpose('loc', 'variable')
    df = pd.DataFrame(data=data.values, columns=features)
    month_length = months_dict[month_query]
    df_monthly = pd.concat([df] * month_length)
    return df_monthly


def filter_bounds(xr_ds, extent):
    """
    Filter xarray bounds
    input: xr_ds: xarray dataset to filter
    input: extent: extent to clip bounds by
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

    cropped_ds = xr_ds.sel(lat=slice(min_lat, max_lat), lon=slice(min_lon, max_lon))

    return cropped_ds

def grab_modis(query_year, geo_extent):
    """
    Grab correct modis array
    """
    # Load appropriate modis data
    modis_ds = xr.open_mfdataset(["/Volumes/MLIA_active_data/data_SUDSAQ/gee_correct/modis/"
                                  "modis_{}_globe_buffersize_55500_with_time.nc".format(query_year)])
    modis_filtered = filter_bounds(modis_ds, geo_extent)
    return modis_filtered

def grab_pop(query_year, geo_extent):
    """
    Grab correct population array
    5-year dataset, choose one closest to date
    """
    # Pop dict, census data must be from within two years
    pop_dict = {
        '2005': ['2005', '2006', '2007'],
        '2010': ['2008', '2009', '2010', '2011', '2012'],
        '2015': ['2013', '2014', '2015', '2016']
    }

    # Load appropriate pop data based on query year
    census_year = None
    for item in pop_dict.keys():
        if query_year in pop_dict[item]:
            census_year = item

    pop_ds = xr.open_mfdataset(["/Volumes/MLIA_active_data/data_SUDSAQ/gee_correct/population/"
                                  "pop_{}_globe_buffersize_55500_with_time.nc".format(census_year)])
    pop_filtered = filter_bounds(pop_ds, geo_extent)
    return pop_filtered

def grab_nightlight(query_year, query_month, band, geo_extent, filepath):
    """
    Grab correct nightlight array
    :query_year: year we are processing data for
    :query_month: month we are processing data for
    :geo_extent: geographic extent of data
    :filepath: location of gee file to process
    """
    nl_ds = xr.open_dataset('{}/nightlight_{}_{}_{}_{}_buffersize_55500_with_time.nc'.format(
        filepath, band, query_month, query_year, geo_extent))
    return nl_ds

nightlight_years = ['2012', '2013', '2014', '2015', '2016']
months = ['aug','july','june']
bands = ['avg_rad', 'cf_cvg']
geo = 'north_america'
save_dir = '/Volumes/PRO-G40/sudsaq/random_forest/nightlight_data_rf/northamerica'
data_dir = '/Volumes/PRO-G40/sudsaq/gee_nightlight_data'

for y in nightlight_years:
    for m in months:
        for b in bands:
            print('Running for nightlight band {} dataset for year: {}, month: {}'.format(b, y, m))
            nightlight = grab_nightlight(y, m, b, geo, data_dir)
            nightlight_features = generate_features(nightlight, 'nightlight', b, m)
            nightlight_features.to_csv('{}/{}_{}_{}_nightlight_features.csv'.format(save_dir, y, m, b))

'''
geo = 'eu' # specify geographic extent here
if geo == 'na':
    geo_name = 'NorthAmerica'
if geo == 'eu':
    geo_name = 'Europe'

save_dir = '/Users/kelseyd/Desktop/random_forest/data/samples/{}/gee'.format(geo_name)

for y in years:
    print('Processing modis features for year: {}'.format(y))
    modis = grab_modis(y, geo)
    for m in months_dict.keys():
        print('Processing for month: {}'.format(m))
        modis_features = generate_features(modis, 'modis', m)
        modis_features.to_csv('{}/{}_{}_modis_features.csv'.format(save_dir, y, m))


for y in years:
    print('Processing population features for year: {}'.format(y))
    pop = grab_pop(y, geo)
    for m in months_dict.keys():
        print('Processing for month: {}'.format(m))
        pop_features = generate_features(pop, 'pop', m)
        pop_features.to_csv('{}/{}_{}_population_features.csv'.format(save_dir, y, m))
'''