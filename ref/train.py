from sklearn.model_selection import train_test_split
import pandas as pd
import time
from time import gmtime, strftime
import numpy as np
import lightgbm as lgb
import os
import gc


print('loading data..')
train_df = pd.read_feather('train_df.ftr')
test_df = pd.read_feather('test_df.ftr')
submit = pd.read_feather('submit.ftr')
y = np.load('y.npy')
print('done')


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

inputs = list(set(train_df.columns) - set([target]))
cat_vars = ['app', 'device', 'os', 'channel', 'hour']

train_df, val_df = train_test_split(train_df, train_size=.95, shuffle=False)
y_train, y_val = train_test_split(y, train_size=.95, shuffle=False)

print('Train size:', len(train_df))
print('Valid size:', len(val_df))

gc.collect()

print('Training...')

num_boost_round = 1000
early_stopping_rounds = 10

xgtrain = lgb.Dataset(train_df[inputs].values, label=y_train,
                      feature_name=inputs,
                      categorical_feature=cat_vars)
del train_df
gc.collect()

xgvalid = lgb.Dataset(val_df[inputs].values, label=y_val,
                      feature_name=inputs,
                      categorical_feature=cat_vars)
del val_df
gc.collect()

evals_results = {}

model = lgb.train(lgb_params,
                  xgtrain,
                  valid_sets=[xgvalid],
                  valid_names=['valid'],
                  evals_result=evals_results,
                  num_boost_round=num_boost_round,
                  early_stopping_rounds=early_stopping_rounds,
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
submit['is_attributed'] = model.predict(test_df[inputs])

print('dump submission')
submit.to_csv('submit.csv', index=False, float_format='%.9f')
print('Done!')

