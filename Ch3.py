import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# Exercise 3.1
dollar_bars = pd.read_csv('data/dollar_bars.csv')
close = dollar_bars['close']
daily_vol = util.get_daily_vol(close)
symm_cusum = filters.cusum_filter(close, threshold=daily_vol)

t1 = labeling.add_vertical_barrier(symm_cusum, close, num_days=1)

# get_events의 side 설정하기 -> triple barrier 0 or 1
first_barrier_touched = labeling.get_events(close, symm_cusum, [1, 1], min_ret=0.03, num_threads=1, t1=t1)
meta_labels = labeling.get_bins(first_barrier_touched, close)


