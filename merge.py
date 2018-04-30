import pandas as pd
import dask.dataframe as dd
from functools import reduce
import gc

print('1. loading...')
train_count_accum = pd.read_feather('train_count_accum.ftr')
train_uniq_nextClick = pd.read_feather('train_uniq_nextClick.ftr')
print('done')

print('2. converting...')
sd1 = dd.from_pandas(train_count_accum, npartitions=3)
sd2 = dd.from_pandas(train_uniq_nextClick, npartitions=3)
print('done')

print('3. dumping...')
sd3 = sd1.merge(sd2, on=['index'])
sd3.to_csv('train_features.csv')
print('done')