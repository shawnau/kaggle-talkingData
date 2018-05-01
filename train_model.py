import numpy as np
import pandas as pd
import lightgbm as lgb
import gc
from time import gmtime, strftime


lgb_params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.05,
    #'is_unbalance': 'true',  #because training data is unbalance (replaced with scale_pos_weight)
    'num_leaves': 31,  # we should let it be smaller than 2^(max_depth)
    'max_depth': -1,  # -1 means no limit
    'min_child_samples': 20,  # Minimum number of data need in a child(min_data_in_leaf)
    'max_bin': 255,  # Number of bucketed bin for feature values
    'subsample': 0.6,  # Subsample ratio of the training instance.
    'subsample_freq': 0,  # frequence of subsample, <=0 means no enable
    'colsample_bytree': 0.3,  # Subsample ratio of columns when constructing each tree.
    'min_child_weight': 5,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
    'subsample_for_bin': 200000,  # Number of samples for constructing bin
    'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
    'reg_alpha': 0,  # L1 regularization term on weights
    'reg_lambda': 0,  # L2 regularization term on weights
    'nthread': 8,
    'verbose': 0,
}


features = [
    'ip', 'app', 'device', 'os', 'channel',
    'day', 'hour', 'app_count', 'ip_count', 'ip_device_count',
    'app_channel_count', 'app_channel_day_count',
    'app_channel_day_hour_count', 'app_countAccum',
    'app_channel_countAccum', 'ip_countAccum',
    'device_channel_day_hour_countAccum', 'app_device_day_hour_countAccum',
    'app_channel_day_hour_countAccum', 'app_uniq_ip_countUniq',
    'app_day_uniq_ip_countUniq', 'app_channel_hour_uniq_os_countUniq',
    'ip_uniq_channel_countUniq', 'ip_uniq_app_countUniq',
    'app_channel_day_hour_uniq_os_countUniq',
    'app_device_channel_uniq_ip_countUniq', 'ip_uniq_hour_countUniq',
    'ip_uniq_os_countUniq', 'app_channel_nextClick', 'ip_nextClick',
    'channel_nextClick', 'ip_device_nextClick', 'channel_day_nextClick',
    'app_nextClick'
]
categorical = ['ip','app', 'device', 'os', 'channel', 'hour']

print('loading data')
xgtrain = lgb.Dataset('data/train_features.bin',
                      feature_name=features,
                      categorical_feature=categorical
                     )
xgvalid = lgb.Dataset('data/valid_features.bin',
                      feature_name=features,
                      categorical_feature=categorical
                     )
print('done')

params = {
    'learning_rate': 0.05,
    #'is_unbalance': 'true', # replaced with scale_pos_weight argument
    'num_leaves': 31,  # 2^max_depth - 1
    'max_depth': -1,  # -1 means no limit
    'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
    'max_bin': 100,  # Number of bucketed bin for feature values
    'subsample': 0.7,  # Subsample ratio of the training instance.
    'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
    'colsample_bytree': 0.9,  # Subsample ratio of columns when constructing each tree.
    'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
    'scale_pos_weight' :200 # because training data is extremely unbalanced
}
lgb_params.update(params)


evals_results = {}
bst = lgb.train(lgb_params, 
                 xgtrain, 
                 valid_sets=[xgvalid], 
                 valid_names=['valid'], 
                 evals_result=evals_results, 
                 num_boost_round=100,
                 early_stopping_rounds=None,
                 verbose_eval=5, 
                 feval=None)


print("\nModel Report")
print("bst1.best_iteration: ", bst.best_iteration)
print('auc'+":", evals_results['valid']['auc'][bst.best_iteration-1])

gain = bst.feature_importance('gain')
ft = pd.DataFrame({'feature':bst.feature_name(), 'split':bst.feature_importance('split'), 'gain':100 * gain / gain.sum()}).sort_values('gain', ascending=False)
ft.to_csv('feature_importance.csv',index=False)
print(ft)


model_name = 'data/model-%s'%strftime("%Y-%m-%d-%H-%M-%S", gmtime())
bst.save_model(model_name)
print('model saved as %s'%model_name)
