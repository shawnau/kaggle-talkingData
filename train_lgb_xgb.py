import gc
from time import gmtime, strftime

from sklearn.model_selection import train_test_split
import lightgbm as lgb
import xgboost as xgb
import numpy as np
import pandas as pd


def group_label(df, group_cols):
    for i, cols in enumerate(group_cols):
        col_name = "_".join(group_cols)
        print(i, col_name)
        group_idx = df.drop_duplicates(cols)[cols].reset_index()
        group_idx.rename(columns={'index': col_name}, inplace=True)
        df = df.merge(group_idx, on=cols, how='left')
        del group_idx
        gc.collect()
    return df


def count_agg(df, group_cols):
    for i, cols in enumerate(group_cols):
        col_name = "_".join(cols) + '_count'
        print(i, col_name)
        count = df.groupby(cols).size().reset_index(name=col_name)
        df = df.merge(count, on=cols, how='left')
        del count
        gc.collect()
    return df


def count_cum(df, group_cols):
    for i, cols in enumerate(group_cols):
        col_name = "_".join(cols) + '_countAccum'
        print(i, col_name)
        df[col_name] = df.groupby(cols).cumcount()
        gc.collect()
    return df


def count_uniq(df, group_uniq_cols):
    for i, cols in enumerate(group_uniq_cols):
        group_cols, uniq_col = cols[0], cols[1]
        col_name = "_".join(group_cols) + '_uniq_' + uniq_col + '_countUniq'
        print(i, col_name)
        tmp = df.groupby(group_cols)[uniq_col].nunique().reset_index(name=col_name)
        df = df.merge(tmp, on=group_cols, how='left')
        del tmp
        gc.collect()
    return df


def next_click(df, group_cols):
    for i, cols in enumerate(group_cols):
        col_name = "_".join(cols) + '_nextClick'
        print(i, col_name)
        df[col_name] = (df.groupby(cols).click_time.shift(-1) - df.click_time).astype(np.float32)
        gc.collect()
    return df


def frequence(df, group_cols):
    for i, cols in enumerate(group_cols):
        col_name = "_".join(cols) + '_nextClick'
        print(i, col_name)
        clickFreq = df.groupby(cols)[col_name].mean().dropna().reset_index(name=("_".join(cols) + '_clickFreq'))
        df = df.merge(clickFreq, on=cols, how='left')
        del clickFreq
        gc.collect()
    return df


def generate_features(df):
    print('generating time features...')
    df['day'] = df['click_time'].dt.day.astype('uint8')
    df['hour'] = df['click_time'].dt.hour.astype('uint8')
    df['in_test_hh'] = (3 - 2 * df['hour'].isin([4, 5, 9, 10, 13, 14])  # most frequent
                        - 1 * df['hour'].isin([6, 11, 15])).astype('uint8')  # least frequent
    print('done')
    gc.collect()

    group_combinations = [
        # ['app', 'device'],
        # ['app', 'channel']
    ]

    count_combinations = [
        ['app'],
        ['ip'],  # 3.03
        ['channel'],
        ['os'],
        ['ip', 'device'],  # 9.88
        ['day', 'hour', 'app'],  # 4.08
        ['app', 'channel'],  # 2.8
        ['ip', 'day', 'in_test_hh'],  # 1.74
        ['ip', 'day', 'hour'],  # 0.52
        ['os', 'device'],  # 0.44
        ['ip', 'os', 'day', 'hour'],  # 0.41
        ['ip', 'device', 'day', 'hour'],  # 0.31
        ['ip', 'app', 'os']  # 0.21
    ]

    countUniq_combinations = [
        # [['app'],'ip'],
        # [['app', 'device', 'os', 'channel'], 'ip'],
        [['ip'], 'channel'],  # 0.9
        [['ip'], 'app'],  # 1.3
        [['ip'], 'os']  # 0.45
    ]

    nextClick_combinations = [
        ['ip', 'os'],
        ['ip', 'device', 'os'],
        ['ip', 'app', 'device', 'os'],
        ['ip', 'app', 'device', 'os', 'channel']
    ]

    freq_combinations = [
        # ['ip', 'app', 'device', 'os']
    ]

    accum_combinations = [
        # ['app'],
        ['ip']  # 3.03
        # ['day', 'hour', 'app']
    ]

    df = group_label(df, group_combinations)
    df = count_agg(df, count_combinations)
    df = count_cum(df, accum_combinations)
    df = count_uniq(df, countUniq_combinations)
    df['click_time'] = (df['click_time'].astype(np.int64) // 10 ** 9).astype(np.int32)
    df = next_click(df, nextClick_combinations)
    df = frequence(df, freq_combinations)

    df.drop(['ip', 'click_time', 'day', 'in_test_hh'], axis=1, inplace=True)
    gc.collect()
    print(df.info())
    return df


# Load data
dtype = {
    'ip' :'uint32',
    'app' :'uint16',
    'device': 'uint16',
    'os' :'uint16',
    'channel': 'uint16',
    'is_attributed': 'uint8',
    'click_id': 'uint32',
}
print('loading train.csv')
# train: (184903890, 7)
# test: (18790469, 7)
train_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed']
train_df = pd.read_csv('data/train.csv', dtype=dtype, usecols=train_cols, parse_dates=['click_time'])
print('loading test.csv')
test_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'click_id']
# using test_supplement
test_df = pd.read_csv('data/test_supplement.csv', dtype=dtype, usecols=test_cols, parse_dates=['click_time'])

# combine train and test data
common_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time']
all_df = pd.concat([train_df[common_cols], test_df[common_cols]])

# generate data
all_df = generate_features(all_df)

# split train/test features from concated data
train_features = all_df.iloc[:train_df.shape[0]]
test_features = all_df.iloc[train_df.shape[0]:]
gc.collect()


########################### train LGB ###########################
lgb_params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.08,
    'num_leaves': 8,
    'max_depth': 4,
    'min_child_samples': 100,
    'max_bin': 100,
    'subsample': 0.7,
    'subsample_freq': 1,
    'colsample_bytree': 0.7,
    'min_child_weight': 0,
    'min_split_gain': 0,
    'nthread': 24,
    'verbose': 1,
    'scale_pos_weight': 200
}

target = 'is_attributed'
features = [col for col in train_features.columns if col not in ['level_0', 'index', 'is_attributed']]
category = ['app', 'device', 'os', 'channel', 'hour']

# train valid split
labels = train_df.is_attributed.values
train_features, valid_features = train_test_split(train_features, test_size=5000000, shuffle=False)
train_labels, valid_labels = train_test_split(labels, test_size=5000000, shuffle=False)
print('Train size:', len(train_features))
print('Valid size:', len(valid_features))
gc.collect()

# convert data into dataset. Warning: Memory Peak
print('converting xgtrain...')
xgtrain = lgb.Dataset(train_features[features].values,
                      label=train_labels,
                      feature_name=features,
                      categorical_feature=category)

print('converting xgvalid...')
xgvalid = lgb.Dataset(valid_features[features].values,
                      label=valid_labels,
                      feature_name=features,
                      categorical_feature=category)

print('Training...')
evals_results = {}
model = lgb.train(lgb_params,
                  xgtrain,
                  valid_sets=[xgvalid],
                  valid_names=['valid'],
                  evals_result=evals_results,
                  num_boost_round=5000,
                  early_stopping_rounds=100,
                  verbose_eval=1,
                  feval=None)
n_estimators = model.best_iteration

print('\nModel Info:')
print('n_estimators:', n_estimators)
print('auc' + ':', evals_results['valid']['auc'][n_estimators - 1])

gain = model.feature_importance('gain')
ft = pd.DataFrame({'feature': model.feature_name(), 'split': model.feature_importance('split'),
                   'gain': 100 * gain / gain.sum()}).sort_values('gain', ascending=False)
ft.to_csv('feature_importance_ref.csv', index=False)
print(ft)

model_name = 'model-%s' % strftime("%Y-%m-%d-%H-%M-%S", gmtime())
model.save_model(model_name)
print('model saved as %s' % model_name)

print('Predicting...')
test_df['is_attributed'] = model.predict(test_features[features], num_iteration=model.best_iteration)

print('loading test')
test = pd.read_csv('data/test.csv', dtype=dtype, usecols=test_cols, parse_dates=['click_time'])

print('merging test_supplement to test')
join_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time']
all_cols = join_cols + ['is_attributed']

test = test.merge(test_df[all_cols], how='left', on=join_cols)
test = test.drop_duplicates(subset=['click_id'])

print("Writing the submission data into a csv file...")
test[['click_id', 'is_attributed']].to_csv('submit_lgb_%s.gz'%(model.best_iteration), index=False, float_format='%.9f', compression='gzip')
print("All done...")

del test
gc.collect()

########################### train XGB ###########################
xgb_params = {'eta': 0.08,
              'tree_method': "hist",
              'grow_policy': "lossguide",
              'max_leaves': 1400,
              'max_depth': 4,
              'subsample': 0.7,
              'colsample_bytree': 0.7,
              'colsample_bylevel':0.7,
              'min_child_weight':0,
              'alpha':0,
              'objective': 'binary:logistic',
              'eval_metric': 'auc',
              'nthread':24,
              'random_state': 42,
              'scale_pos_weight':200,
              'silent': True}

print('converting dtrain...')
dtrain = xgb.DMatrix(train_features, train_labels)
dvalid = xgb.DMatrix(valid_features, valid_labels)
watchlist = [(dvalid, 'valid')]

xgb_model = xgb.train(xgb_params,
                      dtrain,
                      num_boost_round=5000,
                      evals=watchlist,
                      maximize=True,
                      early_stopping_rounds = 100,
                      verbose_eval=5)

model_name = 'xgb-model-%s' % strftime("%Y-%m-%d-%H-%M-%S", gmtime())
xgb_model.save_model(model_name)
print('model saved as %s' % model_name)

dtest = xgb.DMatrix(test_features)
print('Predicting...')
test_df['is_attributed'] = xgb_model.predict(dtest, ntree_limit=xgb_model.best_ntree_limit)
print('loading test')
test = pd.read_csv('data/test.csv', dtype=dtype, usecols=test_cols, parse_dates=['click_time'])

print('merging test_supplement to test')
join_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time']
all_cols = join_cols + ['is_attributed']

test = test.merge(test_df[all_cols], how='left', on=join_cols)
test = test.drop_duplicates(subset=['click_id'])

print("Writing the submission data into a csv file...")
test[['click_id', 'is_attributed']].to_csv('submit_xgb_%s_%s.gz' % (xgb_model.best_ntree_limit, xgb_model.best_score), index=False, float_format='%.9f', compression='gzip')
print("All done...")

