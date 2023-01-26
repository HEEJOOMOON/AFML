import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_white
import data_structures, filters


raw_data = pd.read_csv('data/IVE_tickbidask.txt', names=['Date', 'Time', 'Price', 'Bid', 'Ask', 'Size'])

def read_kibot_ticks(df):
    cols = list(map(str.lower, ['Date', 'Time', 'Price', 'Bid', 'Ask', 'Size']))
    df.columns = cols
    idx = pd.to_datetime(df['date']+df['time'], format='%m/%d/%Y%H:%M:%S')
    df['date_time'] =idx
    df = df.assign(volume=lambda df: df['size'])\
    .assign(dv=lambda df: df['price']*df['size'])\
    .drop(['date','time'], axis=1)\
    .drop_duplicates()

    return df

df = read_kibot_ticks(raw_data)
df = df[['date_time', 'price', 'volume']]
def mad_outlier(y, thres=3.):

    median = np.median(y)
    diff = np.sum((y-median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_dev = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_dev
    return modified_z_score > thres

mad = mad_outlier(df.price.values.reshape(-1,1))
df = df.loc[~mad]

# Exercise 2.4
# Question: How to assign the thresholds
dollar_bars = data_structures.get_dollar_bars(df, threshold=1000000)
# volume_bars = data_structures.get_volume_bars(df, threshold=10000)
# tick_bars = data_structures.get_tick_bars(df, threshold=100)

dollar_bars.index = dollar_bars.date_time

def bollinger_band(price, window=20, fixed=True, bwidth=0.05, n=2):
    df = pd.DataFrame(price)
    df.columns = ['price']
    df['std'] = df['price'].rolling(window=window).std()
    df['rolling'] = df['price'].rolling(window=window).mean()
    df.dropna(inplace=True)
    if fixed == True:
        df['up'] = df['rolling']*(1+bwidth)
        df['down'] = df['rolling']*(1-bwidth)
    else:
        df['up'] = df['rolling']+n*df['std']
        df['down'] = df['rolling']+n*df['std']

    return df


def bb_sampling(bollinger_band):

    mask_df = pd.DataFrame()
    mask_df['up'] = (bollinger_band['price'] < bollinger_band['up'])
    mask_df['down'] = (bollinger_band['price'] > bollinger_band['down'])
    mask_df['up_next'] = mask_df['up'].shift(-1)
    mask_df['down_next'] = mask_df['down'].shift(-1)
    mask_df.dropna(inplace=True)
    sample_up = mask_df['up'] & mask_df['up_next']
    sample_down = mask_df['down'] & mask_df['down_next']
    down_sample = sample_down.loc[sample_down == False]
    up_sample = sample_up.loc[sample_up == False]


    return up_sample.index, down_sample.index


def cusum_filter_one_way(raw_time_series, threshold, time_stamps=True, pos=True):

    t_events = []
    s_pos = 0

    # log returns
    raw_time_series = pd.DataFrame(raw_time_series)  # Convert to DataFrame
    raw_time_series.columns = ['price']
    raw_time_series['log_ret'] = raw_time_series.price.apply(np.log).diff()
    if isinstance(threshold, (float, int)):
        raw_time_series['threshold'] = threshold
    elif isinstance(threshold, pd.Series):
        raw_time_series.loc[threshold.index, 'threshold'] = threshold
    else:
        raise ValueError('threshold is neither float nor pd.Series!')

    raw_time_series = raw_time_series.iloc[1:]  # Drop first na values

    # Get event time stamps for the entire series

    if pos:
        for tup in raw_time_series.itertuples():
            thresh = tup.threshold
            pos = float(s_pos + tup.log_ret)
            s_pos = max(0.0, pos)

            if s_pos > thresh:
                s_pos = 0
                t_events.append(tup.Index)

        # Return DatetimeIndex or list
        if time_stamps:
            event_timestamps = pd.DatetimeIndex(t_events)
            return event_timestamps

    else:
        for tup in raw_time_series.itertuples():
            thresh = tup.threshold
            pos = float(s_pos + tup.log_ret)
            s_pos = max(0.0, pos)

            if s_pos < -thresh:
                s_pos = 0
                t_events.append(tup.Index)

        # Return DatetimeIndex or list
        if time_stamps:
            event_timestamps = pd.DatetimeIndex(t_events)
            return event_timestamps

    return t_events

bd_df = bollinger_band(dollar_bars['close'])
up_sample, down_sample = bb_sampling(bd_df)
bb_samples = dollar_bars.close.loc[up_sample | down_sample]

plt.plot(bd_df['price'])
plt.plot(bd_df['up'], '--r')
plt.plot(bd_df['down'], '--b')
plt.scatter(bb_samples.index, bb_samples.values)
plt.show()


print('How many times prices cross the bands out: ', len(bb_samples))

cusum = cusum_filter_one_way(dollar_bars['close'], threshold=0.05)
cusum_samples = dollar_bars[['close']].loc[cusum]
print('How many samples are selected by cusum filter: ', len(cusum_samples))


# 이분산성 검정
rolled_std_bb = bb_samples.rolling(20).std().dropna()
rolled_std_cusum = cusum_samples.rolling(20).std().dropna()


# Exercise 2.5
ab_cusum = filters.cusum_filter(dollar_bars['close'], threshold=0.05)
ab_cusum_sampled = dollar_bars[['close']].loc[ab_cusum]

rolled_std_cusum_abs = ab_cusum_sampled.rolling(20).std().dropna()


plt.scatter(rolled_std_bb.index, rolled_std_bb.values)
plt.show()
plt.scatter(rolled_std_cusum.index, rolled_std_cusum.values)
plt.show()
plt.scatter(rolled_std_cusum_abs.index, rolled_std_cusum_abs.values)
plt.show()