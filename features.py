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
df = pd.read_csv('data/train.csv', dtype=dtype, usecols=dtype.keys(), parse_dates=['click_time'])
print('Done')

# times
print('time features...')
df['click_time']= pd.to_datetime(df['click_time'])
df = df.sort_values(by=['click_time'])
df['day'] = df['click_time'].dt.day.astype('uint8')
df['hour'] = df['click_time'].dt.hour.astype('uint8')
df['click_time'] = (df['click_time'].astype(np.int64) // 10 ** 9).astype(np.int32)
print('Done')

# train/valid
df = df[df.day != 9]
#df = df[df.day == 9]
print('dataset: ', df.shape)
gc.collect()

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
    # ['app', 'channel', 'os']，
    # ['device', 'os']，
    # ['ip', 'os', 'day', 'hour']，
    # ['ip', 'app', 'day', 'hour']，
    # ['ip', 'device', 'day', 'hour']，
    # ['ip', 'app', 'os']，
    # ['app', 'day', 'hour']，
    ['ip', 'day', 'hour'],
    ['ip', 'app'],
    ['app'], # 7.2
    ['ip'],  # 3.1
    ['ip', 'device'], # 0.53
    ['app', 'channel'], # 2.6
    # ['app', 'channel', 'day'], # 0.08
    ['app', 'channel', 'day', 'hour'] # 0.68
]
for i, cols in enumerate(count_combinations):
    print(i, cols)
    df = count_agg(df, cols)


# accumulate count agg features
countAccum_combinations = [
    ['ip', 'device', 'os'],
    ['app'], # 0.088
    #['app', 'channel'], # 0.045
    ['ip'] # 0.081
    # ['device', 'channel', 'day', 'hour'], # 0.02
    # ['app', 'device', 'day', 'hour'], # 0.04
    # ['app', 'channel', 'day', 'hour'] # 0.03
]
for i, cols in enumerate(countAccum_combinations):
    print(i, cols)
    df = count_cum(df, cols)


# unique count agg features
countUniq_combinations = [
    # [['ip', 'device', 'os'], 'app'],
    # [['ip', 'day'], 'hour'],
    # [['ip'], 'device'],
    # [['app'], 'channel'],
    # [['app'], 'ip'], # 0.09
    [['app', 'day'], 'ip'], # 3
    [['app', 'channel', 'hour'], 'os'], # 2
    [['ip'], 'channel'], # 0.9
    [['ip'], 'app'], # 1.3
    [['app', 'channel', 'day', 'hour'], 'os'], # 9.31
    # [['app', 'device', 'channel'], 'ip'], # 0.06
    # [['ip'], 'hour'], # 0.08
    [['ip'], 'os'] # 0.45
]
for i, cols in enumerate(countUniq_combinations):
    print(i, cols)
    df = count_uniq(df, cols[0], cols[1])


# next click features
next_click_combinations = [
    ['ip', 'app', 'device', 'os', 'channel'],
    ['ip', 'os', 'device'],
    ['ip', 'os', 'device', 'app'],
    ['app', 'channel'], # 3.3
    ['ip'], # 1.8
    # ['channel'], # 1.3 0.000978
    ['ip', 'device'] # 0.75
    # ['channel', 'day'], # 0.61 0.001321
    # ['app'] # 0.41 0.000459
]
for i, cols in enumerate(next_click_combinations):
    print(i, cols)
    df = next_click(df, cols)

del df['click_time']
gc.collect()

print('dumping...')
df.to_feather('data/train_features.ftr')
print('done')
