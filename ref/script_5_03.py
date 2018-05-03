import gc
import os
import time
from time import gmtime, strftime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb


def count_agg(df, group_cols):
    col_name = "_".join(group_cols) + '_count'
    count = df.groupby(group_cols).size().reset_index(name=col_name)
    df = df.merge(count, on=group_cols, how='left')
    del count
    gc.collect()
    return df


def count_cum(df, group_cols):
    col_name = "_".join(group_cols) + '_countAccum'
    df[col_name] = df.groupby(group_cols).cumcount()
    gc.collect()
    return df


def count_uniq(df, group_cols, uniq_col):
    col_name = "_".join(group_cols) + '_uniq_' + uniq_col + '_countUniq'
    tmp = df.groupby(group_cols)[uniq_col].nunique().reset_index(name=col_name)
    df = df.merge(tmp, on=group_cols, how='left')
    del tmp
    gc.collect()
    return df


def next_click(df, group_cols):
    df["_".join(group_cols) + '_nextClick'] = (df.groupby(group_cols).click_time.shift(-1) - df.click_time).astype(
        np.float32)
    gc.collect()
    return df


def generate_features(df):
    print('generating time features...')
    df['day'] = df['click_time'].dt.day.astype('uint8')
    df['hour'] = df['click_time'].dt.hour.astype('uint8')
    df['in_test_hh'] = (3
                        - 2 * df['hour'].isin([4, 5, 9, 10, 13, 14])  # most frequent
                        - 1 * df['hour'].isin([6, 11, 15])).astype('uint8')  # least frequent
    print('done')
    gc.collect()

    count_combinations = [
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

    # count features
    for i, cols in enumerate(count_combinations):
        print(i, cols)
        df = count_agg(df, cols)

    # next click features
    df['click_time'] = (df['click_time'].astype(np.int64) // 10 ** 9).astype(np.int32)
    for i, cols in enumerate(nextClick_combinations):
        print(i, cols)
        df = next_click(df, cols)

    df.drop(['ip', 'day', 'click_time', 'in_test_hh'], axis=1, inplace=True)
    gc.collect()
    print(df.info())


# load data
dtype = {
    'ip': 'uint32',
    'app': 'uint16',
    'device': 'uint16',
    'os': 'uint16',
    'channel': 'uint16',
    'is_attributed': 'uint8',
    'click_id': 'uint32',
}

# train: (184903890, 7)
# test: (18790469, 7)
print('load train.csv')
train_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed']
train_df = pd.read_csv('../data/train.csv', skiprows=range(1, 84903891), nrows=100000000,
                       dtype=dtype, usecols=train_cols, parse_dates=['click_time'])
print('load test.csv')
test_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'click_id']
test_df = pd.read_csv('../data/test.csv', dtype=dtype, usecols=test_cols, parse_dates=['click_time'])

# combine train and test data
common_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time']
all_df = pd.concat([train_df[common_cols], test_df[common_cols]])

# generate features
all_df = generate_features(all_df)

# cache redundant cols
submit = pd.DataFrame()
submit['click_id'] = test_df['click_id']
y = train_df.is_attributed.values

# split up into train and test
train_df = train_df.iloc[:train_df.shape[0]]
test_df = train_df.iloc[train_df.shape[0]:]
gc.collect()

metrics = 'auc'
lgb_params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': metrics,
    'learning_rate': 0.1,
    'num_leaves': 7,
    'max_depth': 4,
    'min_child_samples': 100,
    'max_bin': 100,
    'subsample': 0.7,
    'subsample_freq': 1,
    'colsample_bytree': 0.7,
    'min_child_weight': 0,
    'min_split_gain': 0,
    'nthread': 4,
    'verbose': 1,
    'scale_pos_weight': 99.7
    # 'scale_pos_weight': 400
}

target = 'is_attributed'
features = [col for col in train_df.columns if col not in ['is_attributed']]
category = ['app', 'device', 'os', 'channel', 'hour']

# train valid split
train_df, val_df = train_test_split(train_df, train_size=.95, shuffle=False)
y_train, y_val = train_test_split(y, train_size=.95, shuffle=False)
print('Train size:', len(train_df))
print('Valid size:', len(val_df))
gc.collect()

# convert data into dataset
xgtrain = lgb.Dataset(train_df[features].values, label=y_train,
                      feature_name=features,
                      categorical_feature=category)
del train_df
gc.collect()

xgvalid = lgb.Dataset(val_df[features].values, label=y_val,
                      feature_name=features,
                      categorical_feature=category)
del val_df
gc.collect()

print('Training...')
evals_results = {}
model = lgb.train(lgb_params,
                  xgtrain,
                  valid_sets=[xgvalid],
                  valid_names=['valid'],
                  evals_result=evals_results,
                  num_boost_round=1000,
                  early_stopping_rounds=20,
                  verbose_eval=1,
                  feval=None)
n_estimators = model.best_iteration

print('\nModel Info:')
print('n_estimators:', n_estimators)
print(metrics + ':', evals_results['valid'][metrics][n_estimators - 1])

del xgvalid
del xgtrain
gc.collect()

gain = model.feature_importance('gain')
ft = pd.DataFrame({'feature': model.feature_name(), 'split': model.feature_importance('split'),
                   'gain': 100 * gain / gain.sum()}).sort_values('gain', ascending=False)
ft.to_csv('feature_importance_ref.csv', index=False)
print(ft)

model_name = 'model-%s' % strftime("%Y-%m-%d-%H-%M-%S", gmtime())
model.save_model(model_name)
print('model saved as %s' % model_name)

print('Predicting...')
submit['is_attributed'] = model.predict(test_df[features])

print('dump submission')
submit.to_csv('submit.csv', index=False, float_format='%.9f')
print('Done!')