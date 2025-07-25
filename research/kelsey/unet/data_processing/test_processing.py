import xarray as xr
import os
import numpy as np
import datetime as dt


# Top ten features from July RF experiments
feature_list = ['momo.2dsfc.NH3', 'momo.2dsfc.PROD.HOX', 'momo.2dsfc.DMS', 'momo.co',
                'momo.2dsfc.NALD', 'momo.2dsfc.HNO3', 'momo.2dsfc.BrONO2', 'momo.t',
                'momo.no2', 'momo.2dsfc.PAN']

Timezones = [
    (0, (0.0, 7.5)),
    (1, (7.5, 22.5)),
    (2, (22.5, 37.5)),
    (3, (37.5, 52.5)),
    (4, (52.5, 67.5)),
    (5, (67.5, 82.5)),
    (6, (82.5, 97.5)),
    (7, (97.5, 112.5)),
    (8, (112.5, 127.5)),
    (9, (127.5, 142.5)),
    (10, (142.5, 157.5)),
    (11, (157.5, 172.5)),
    (12, (172.5, 180.0)),
    (-12, (-180.0, -172.5)),
    (-11, (-172.5, -157.5)),
    (-10, (-157.5, -142.5)),
    (-9, (-142.5, -127.5)),
    (-8, (-127.5, -112.5)),
    (-7, (-112.5, -97.5)),
    (-6, (-97.5, -82.5)),
    (-5, (-82.5, -67.5)),
    (-4, (-67.5, -52.5)),
    (-3, (-52.5, -37.5)),
    (-2, (-37.5, -22.5)),
    (-1, (-22.5, -7.5)),
    (0, (-7.5, 0.0))
]

def daily(ds):
    """
    Aligns a dataset to a daily average
    """
    def select_times(ds, sel, time):
        """
        Selects timestamps using integer hours (0-23) over all dates
        """
        if isinstance(sel, list):
            mask = (dt.time(sel[0]) <= time) & (time <= dt.time(sel[1]))
            ds   = ds.where(mask, drop=True)
            # Floor the times to the day for the groupby operation
            ds.coords['time'] = ds.time.dt.floor('1D')

            # Now group as daily taking the mean
            ds = ds.groupby('time').mean()
        else:
            mask = (time == dt.time(sel))
            ds   = ds.where(mask, drop=True)

            # Floor the times to the day for the groupby operation
            ds.coords['time'] = ds.time.dt.floor('1D')

        return ds

    # Select time ranges per config
    time = ds.time.dt.time
    data = []
    time_range = [8, 15]
    print('-- Using local timezones')
    ns = ds[feature_list]
    local = []
    for offset, bounds in Timezones:
        print('Running bounds: {}'.format(bounds))
        sub  = ns.sel(lon=slice(*bounds))
        time = ( sub.time + np.timedelta64(offset, 'h') ).dt.time
        sub  = select_times(sub, time_range, time)
        local.append(sub)

    # Merge these datasets back together to create the full grid
    print('-- Merging local averages together (this can take awhile)')
    data.append(xr.merge(local))
    # Add variables that don't have a time dimension back in
    timeless = ds.drop_dims('time')
    if len(timeless) > 0:
        print(f'- Appending {len(timeless)} timeless variables: {list(timeless)}')
        data.append(timeless)

    # Merge the selections together
    print('- Merging all averages together')
    ds = xr.merge(data)

    # Cast back to custom Dataset (xr.merge creates new)
    ds = xr.Dataset(ds)

    return ds

def filter_bounds(xr_ds):
    """
    Filter xarray bounds (currently hardcoded for NA)
    """
    min_lat = 20.748
    max_lat = 54.392
    min_lon = -124.875
    max_lon = -73.375
    cropped_ds = xr_ds.sel(lat=slice(min_lat, max_lat), lon=slice(min_lon, max_lon))

    return cropped_ds

def format_lon(x):
    """
    Format ds longitude to range -180 - 180
    """
    if int(x.coords['lon'].max()) > 180:
        x.coords['lon'] = (x.coords['lon'] + 180) % 360 - 180
    return x.sortby(x.lon)


test = xr.open_mfdataset(["/Volumes/MLIA_active_data/data_SUDSAQ/data/momo/2005/08.nc"])
filtered = test[feature_list]
filt_lon = format_lon(filtered)
na_ds = filter_bounds(filt_lon)
daily_ds = daily(na_ds)
import ipdb
ipdb.set_trace()