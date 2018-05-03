from sklearn.model_selection import train_test_split
import pandas as pd
import time
from time import gmtime, strftime
import numpy as np
import lightgbm as lgb
import os


output_file = 'lgbm_submit.csv'

path = "../data/"

dtypes = {
    'ip' :'uint32',
    'app' :'uint16',
    'device': 'uint16',
    'os' :'uint16',
    'channel': 'uint16',
    'is_attributed': 'uint8',
    'click_id': 'uint32',
}

print('Loading train.csv...')

train_cols = ['ip', 'app', 'device', 'os', 'channel', 'is_attributed', 'click_time']
train_df = pd.read_csv(path + 'train.csv', skiprows=range(1, 84903891), nrows=100000000, dtype=dtypes,
                       usecols=train_cols)

print('Loading test.csv...')
test_cols = ['ip', 'app', 'device', 'os', 'click_time', 'channel', 'click_id']
test_df = pd.read_csv(path + "test.csv", dtype=dtypes, usecols=test_cols)

import gc

len_train = len(train_df)

print('Preprocessing...')

agg_combinations = [
    ['ip'],
    ['os', 'device'],
    ['os', 'app', 'channel'],
    ['ip', 'device'],
    ['app', 'channel'],
    ['ip', 'day', 'in_test_hh'],
    ['ip', 'day', 'hour'],
    ['ip', 'os', 'day', 'hour'],
    ['ip', 'app', 'day', 'hour'],
    ['ip', 'device', 'day', 'hour'],
    ['ip', 'app', 'os'],
    ['day', 'hour', 'app']
]

nextClick_combinations = [
    ['ip', 'app', 'device', 'os']
]

def add_counts(df, cols):
    print('add: ', cols)
    arr_slice = df[cols].values
    unq, unqtags, counts = np.unique(np.ravel_multi_index(arr_slice.T, arr_slice.max(axis=0) + 1),
                                     return_inverse=True, return_counts=True)
    df["_".join(cols) + "_count"] = counts[unqtags]


def add_next_click(df, group_cols):
    df["_".join(group_cols)+'_nextClick'] = (df.groupby(group_cols).click_time.shift(-1) - df.click_time).astype(np.float32)
    gc.collect()


def preproc_data(df):
    # Extrace date info
    df['click_time'] = pd.to_datetime(df['click_time'])
    df['hour'] = df['click_time'].dt.hour.astype('uint8')
    df['day'] = df['click_time'].dt.day.astype('uint8')
    gc.collect()

    # Groups
    most_freq_hours_in_test_data = [4, 5, 9, 10, 13, 14]
    least_freq_hours_in_test_data = [6, 11, 15]
    df['in_test_hh'] = (3
                        - 2 * df['hour'].isin(most_freq_hours_in_test_data)
                        - 1 * df['hour'].isin(least_freq_hours_in_test_data)).astype('uint8')

    print('Count Grouping...')
    for cols in agg_combinations:
        add_counts(df, cols)
    
    print('Adding next_click...')
    df['click_time'] = (df['click_time'].astype(np.int64) // 10 ** 9).astype(np.int32)
    for cols in nextClick_combinations:
        add_next_click(df, cols)
    
    df.drop(['ip', 'day', 'click_time', 'in_test_hh'], axis=1, inplace=True)
    gc.collect()

    print(df.info())

    return df


y = train_df.is_attributed.values

submit = pd.DataFrame()
submit['click_id'] = test_df['click_id']

train_len = len(train_df)
common_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time']
train_df = pd.concat([train_df[common_cols], test_df[common_cols]])

train_df = preproc_data(train_df)

test_df = train_df.iloc[train_len:]
train_df = train_df.iloc[:train_len]

gc.collect()

print('dumping data..')
train_df.to_feather('train_df.ftr')
test_df.to_feather('test_df.ftr')
submit.to_feather('submit.ftr')
np.save('y.npy', y)
print('done')