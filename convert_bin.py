import numpy as np
import pandas as pd
import lightgbm as lgb
import gc

print('loading data...')
train = pd.read_feather('data/train_features.ftr')
valid = pd.read_feather('data/valid_features.ftr')

print('train: ', train.shape, 'valid: ', valid.shape)
print('features: ', train.columns)

features = [x for x in train.columns if x not in ['index', 'click_time', 'is_attributed']]
categorical = ['ip', 'app', 'device', 'os', 'channel', 'hour']
respond = 'is_attributed'

print('converting...')
xgtrain = lgb.Dataset(train[features].values, 
                      label=train[respond].values,
                      feature_name=features,
                      categorical_feature=categorical
                     )

xgvalid = xgtrain.create_valid(valid[features].values, 
                               label=valid[respond].values)
# bug?
xgvalid.feature_name = features
xgvalid.categorical_feature = categorical


print('dump to bin...')
xgtrain.save_binary('data/train_features.bin')
xgvalid.save_binary('data/valid_features.bin')
print('done')