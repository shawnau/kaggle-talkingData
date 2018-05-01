import itertools
import numpy as np
import pandas as pd
import gc

dtype = {
    'ip': np.int32,
    'app': np.int16,
    'device': np.int16,
    'os': np.int16,
    'channel': np.int16,
    'click_time': object,
    'is_attributed': np.int16
}

print('loading train data...')
df = pd.read_csv('train_sample.csv', dtype=dtype, usecols=dtype.keys(), parse_dates=['click_time'])
print('Done')

# times
print('time features...')
df['click_time']= pd.to_datetime(df['click_time'])
df = df.sort_values(by=['click_time'])
df['day'] = df['click_time'].dt.day.astype('uint8')
df['hour'] = df['click_time'].dt.hour.astype('uint8')
df['click_time'] = (df['click_time'].astype(np.int64) // 10 ** 9).astype(np.int32)
print('Done')

# features
def count_agg(df, group_cols):
    col_name = "_".join(group_cols)+'_count'
    count = df.groupby(group_cols).size().reset_index(name=col_name)
    df = df.merge(count, on=group_cols, how='left')
    del count
    gc.collect()
    return df

def count_cum(df, group_cols):
    col_name = "_".join(group_cols)+'_countAccum'
    df[col_name] = df.groupby(group_cols).cumcount()
    gc.collect()
    return df

def count_uniq(df, group_cols, uniq_col):
    col_name = "_".join(group_cols)+'_uniq_'+uniq_col+'_countUniq'
    tmp = df.groupby(group_cols)[uniq_col].nunique().reset_index(name=col_name)
    df = df.merge(tmp, on=group_cols, how='left')
    del tmp
    gc.collect()
    return df

def next_click(df, group_cols):
    df["_".join(group_cols)+'_nextClick'] = (df.groupby(group_cols).click_time.shift(-1) - df.click_time).astype(np.float32)
    gc.collect()
    return df


# count agg features
count_combinations = [
    ['app'], # 37
    ['ip'],  # 4
    ['ip', 'device'], # 0.53
    ['app', 'channel'], # 0.51
    ['app', 'channel', 'day'], # 0.50
    ['app', 'channel', 'day', 'hour'] # 0.33
]
for i, cols in enumerate(count_combinations):
    print(i, cols)
    df = count_agg(df, cols)


# accumulate count agg features
countAccum_combinations = [
    ['app'], # 7
    ['app', 'channel'], # 3.5
    ['ip'], # 2.5
    ['device', 'channel', 'day', 'hour'], # 0.59
    ['app', 'device', 'day', 'hour'], # 0.39
    ['app', 'channel', 'day', 'hour'] # 0.34
]
for i, cols in enumerate(countAccum_combinations):
    print(i, cols)
    df = count_cum(df, cols)


# unique count agg features
countUniq_combinations = [
    [['app'], 'ip'], # 10
    [['app', 'day'], 'ip'], # 3
    [['app', 'channel', 'hour'], 'os'], # 2
    [['ip'], 'channel'], # 1.8
    [['ip'], 'app'], # 1.5
    [['app', 'channel', 'day', 'hour'], 'os'], # 1.1
    [['app', 'device', 'channel'], 'ip'], # 0.77
    [['ip'], 'hour'], # 0.71
    [['ip'], 'os'] # 0.45
]
for i, cols in enumerate(countUniq_combinations):
    print(i, cols)
    df = count_uniq(df, cols[0], cols[1])


# next click features
next_click_combinations = [
    ['app', 'channel'], # 3.3
    ['ip'], # 1.8
    ['channel'], # 1.3
    ['ip', 'device'], # 0.75
    ['channel', 'day'], # 0.61
    ['app'] # 0.41
]
for i, cols in enumerate(next_click_combinations):
    print(i, cols)
    df = next_click(df, cols)

print('dumping...')
df.to_feather('train_features.ftr')
print('done')
