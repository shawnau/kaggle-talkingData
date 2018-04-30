import itertools
import numpy as np
import pandas as pd

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
df = pd.read_csv('train.csv', dtype=dtype, usecols=dtype.keys(), parse_dates=['click_time'])
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
    return df

def count_cum(df, group_cols):
    col_name = "_".join(group_cols)+'_countAccum'
    df[col_name] = df.groupby(group_cols).cumcount()
    return df

def count_uniq(df, group_cols, uniq_col):
    col_name = "_".join(group_cols)+'_uniq_'+uniq_col+'_countUniq'
    tmp = df.groupby(group_cols)[uniq_col].nunique().reset_index(name=col_name)
    df = df.merge(tmp, on=group_cols, how='left')
    return df

def next_click(df, group_cols):
    df["_".join(group_cols)+'_nextClick'] = (df.groupby(group_cols).click_time.shift(-1) - df.click_time).astype(np.float32)
    return df


# count agg features
count_combinations = [
    ['app'],
    ['ip'],
    ['app', 'channel'],
    ['ip', 'device'],
    ['ip', 'day'],
    ['app', 'channel', 'hour'],
    ['app', 'channel', 'day'],
    ['app', 'channel', 'day', 'hour']
]
for i, cols in enumerate(count_combinations):
    print(i, cols)
    df = count_agg(df, cols)


# accumulate count agg features
countAccum_combinations = [
    ['ip'],
    ['channel'],
    ['app'],
    ['device'],
    ['app', 'channel'],
    ['app', 'channel', 'day'],
    ['channel', 'day', 'hour'],
    ['device', 'channel', 'day', 'hour'],
    ['app', 'channel', 'day', 'hour'],
    ['app', 'device', 'channel', 'day', 'hour'],
    ['ip', 'day'],
    ['ip', 'device']
]
for i, cols in enumerate(countAccum_combinations):
    print(i, cols)
    df = count_cum(df, cols)


# unique count agg features
countUniq_combinations = [
    [['app'], 'ip'],
    [['app', 'day'], 'ip'],
    [['app', 'device', 'channel'], 'ip'],
    [['app', 'hour', 'channel'], 'ip'],
    [['ip'], 'channel'],
    [['ip'], 'app'],
    [['ip'], 'hour'],
    [['ip'], 'os'],
    [['app', 'channel', 'hour'], 'os'],
    [['app', 'channel', 'day', 'hour'], 'os'],
]
for i, cols in enumerate(countUniq_combinations):
    print(i, cols)
    df = count_uniq(df, cols[0], cols[1])


# next click features
next_click_combinations = [
    ['ip'],
    ['channel'],
    ['ip', 'device'],
    ['channel', 'day'],
    ['app', 'channel'],
    ['ip', 'app'],
    ['ip', 'app', 'os'],
    ['ip', 'app', 'os', 'device'],
    ['ip', 'app', 'os', 'device', 'channel'],
]
for i, cols in enumerate(next_click_combinations):
    print(i, cols)
    df = next_click(df, cols)

df.to_feather('train_features.ftr')
