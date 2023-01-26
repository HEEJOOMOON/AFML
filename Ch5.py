import pandas as pd
import numpy as np
from feature_preprocessing.risk_estimators import RiskEstimators
from feature_preprocessing.fracdiff import frac_diff_ffd
from feature_importance import get_orthogonal_features
from statsmodels.tsa.stattools import adfuller
import talib as ta
import networkx as nx

def denoised(df: pd.DataFrame,
             detoned: bool=True,
             corr: bool=False):

    df_cov = df.cov()
    tn = len(df)/len(df.columns)
    cov = RiskEstimators().denoise_covariance(cov=df_cov, tn_relation=tn, detone=detoned)
    cov = pd.DataFrame(cov, index=df_cov.index, columns=df_cov.columns)
    if corr:
        cov = RiskEstimators().cov_to_corr(cov)
    return cov

def detoned_corr(corr, market_component=1):

    # Calculating eigenvalues and eigenvectors of the de-noised matrix

    eigenvalues, eigenvectors = np.linalg.eigh(corr)

    # Index to sort eigenvalues in descending order
    indices = eigenvalues.argsort()[::-1]

    # Sorting
    eigenvalues = eigenvalues[indices]
    eigenvectors = eigenvectors[:, indices]

    # Outputting eigenvalues on the main diagonal of a matrix
    eigenvalues = np.diagflat(eigenvalues)


    # Getting the eigenvalues and eigenvectors related to market component
    eigenvalues_mark = eigenvalues[:market_component, :market_component]
    eigenvectors_mark = eigenvectors[:, :market_component]

    # Calculating the market component correlation
    corr_mark = np.dot(eigenvectors_mark, eigenvalues_mark).dot(eigenvectors_mark.T)

    # Removing the market component from the de-noised correlation matrix
    corr = corr - corr_mark

    # Rescaling the correlation matrix to have 1s on the main diagonal
    corr = RiskEstimators.cov_to_corr(corr)
    return corr


def min_frac_diff(series):

    results = pd.DataFrame(columns=['adfStat', 'pVal', 'lags', 'nObs', '95% conf', 'corr'])

    for d_value in np.linspace(0, 1, 11):
        close_prices = np.log(series[['close']]).resample('1D').last()  # Downcast to daily obs
        close_prices.dropna(inplace=True)

        differenced_series = frac_diff_ffd(close_prices[['close']], diff_amt=d_value, thresh=0.01).dropna()

        corr = np.corrcoef(close_prices.loc[differenced_series.index, 'close'],
                           differenced_series['close'])[0, 1]

        differenced_series = adfuller(differenced_series['close'], maxlag=1, regression='c', autolag=None)

        # Results to dataframe
        results.loc[d_value] = list(differenced_series[:4]) + [differenced_series[4]['5%']] + [corr]  # With critical value

    return results


def minimum_d(series, threshold):
    tmp = pd.DataFrame(series)
    tmp.columns = ['close']
    out = min_frac_diff(tmp)
    min_d = min((out.loc[(out.pVal < threshold)]).index)
    return min_d


def frac_diff(df: pd.Series,
              threshold: float=0.05):

    log_df = pd.DataFrame(np.log(df))
    frac_diff_return = frac_diff_ffd(log_df, diff_amt=minimum_d(df, threshold=threshold), thresh=1e-4)
    return frac_diff_return



def get_mom_st(df: pd.DataFrame):


    df = df.dropna()
    close = df.Close
    high = df.High
    low = df.Low

    mom_st = pd.DataFrame()
    mom_st['RSI_st'] = ta.RSI(close, timeperiod=10)
    mom_st['MACD_st'] = ta.MACD(close, fastperiod=10, slowperiod=20, signalperiod=5)[0]
    mom_st['WILLR_st'] = ta.WILLR(high=high, low=low, close=close, timeperiod=10)
    mom_st['CCI_st'] = ta.CCI(high=high, low=low, close=close, timeperiod=10)
    mom_st['MOM_st'] = ta.MOM(close, timeperiod=10)

    return mom_st


def get_mom_lt(df: pd.DataFrame):

    df = df.dropna()
    close = df.Close
    high = df.High
    low = df.Low

    mom_lt = pd.DataFrame()
    mom_lt['RSI_lt'] = ta.RSI(close, timeperiod=60)
    mom_lt['MACD_lt'] = ta.MACD(close, fastperiod=20, slowperiod=60, signalperiod=20)[0]
    mom_lt['WILLR_lt'] = ta.WILLR(high=high, low=low, close=close, timeperiod=60)
    mom_lt['CCI_lt'] = ta.CCI(high=high, low=low, close=close, timeperiod=60)
    mom_lt['MOM_lt'] = ta.MOM(close, timeperiod=60)

    return mom_lt


def get_ma(df: pd.Series):
    df = df.dropna()
    ma = pd.DataFrame()
    ma['ma5'] = df.rolling(window=5).mean()
    ma['ma10'] = df.rolling(window=10).mean()
    ma['ma20'] = df.rolling(window=20).mean()
    ma['ma60'] = df.rolling(window=60).mean()
    return ma


def get_vol(df: pd.DataFrame):

    df = df.dropna()
    high = df.High
    low = df.Low
    close = df.Close
    volume = df.Volume
    vol = pd.DataFrame()
    vol['ATR_st'] = ta.ATR(high=high, low=low, close=close, timeperiod=10)
    vol['ATR_lt'] = ta.ATR(high=high, low=low, close=close, timeperiod=60)
    vol['ms5'] = close.rolling(window=5).std()
    vol['ms10'] = close.rolling(window=10).std()
    vol['ms20'] = close.rolling(window=20).std()
    vol['ms60'] = close.rolling(window=60).std()
    # vol['OBV'] = ta.OBV(close, volume)
    return vol


def downlaod_data(ls: list, start, end):

    out = pd.DataFrame()
    for l in ls:
        tmp = pd.read_csv(f"./Data/{l}.csv", index_col=0)
        try:
            tmp = tmp.Close
        except AttributeError:
            tmp = tmp[l]
        tmp.index = pd.to_datetime(tmp.index)
        tmp = tmp.sort_index(ascending=True)
        out[l] = tmp.loc[start:end]
    return out


def centrality_rank(measure):
    dict_ = eval(measure)
    dict_new = sorted(dict_.items(), reverse=True, key=lambda x: x[1])
    return dict_new


def get_important_feature(features_df, clusters, method):


    out = pd.Series()
    for c in clusters:

        corr = features_df.loc[:, c].corr()
        graph_corr = corr.stack().reset_index()
        graph_corr.columns = ['node1', 'node2', 'corr_']
        graph_corr = graph_corr[graph_corr.node1 != graph_corr.node2]

        corr_u = pd.DataFrame(pd.np.triu(corr.values), columns=corr.columns, index=corr.index)
        graph_corr_u = corr_u.stack().reset_index()
        graph_corr_u.columns = ['node1', 'node2', 'corr_']
        graph_corr_u = graph_corr_u[graph_corr_u.node1 != graph_corr_u.node2]
        graph_corr_u = graph_corr_u[graph_corr_u.corr_ != 0]

        if method == 'nx.closeness_centrality' or method == 'nx.betweenness_centrality':

            edge_list = [(v['node1'], v['node2'], abs(1/v['corr_'])) for u, v in graph_corr_u.iterrows()]

        else:
            edge_list = [(v['node1'], v['node2'], abs(v['corr_'])) for u, v in graph_corr_u.iterrows()]

        G = nx.from_pandas_edgelist(graph_corr, 'node1', 'node2', create_using=nx.Graph())
        G.add_weighted_edges_from(edge_list, weight='weight')

        if method == 'nx.closeness_centrality':
            dict_ = eval(f"{method}(G, distance='weight')")
        else:
            dict_ = eval(f"{method}(G, weight='weight')")

        out = out.append(pd.Series(dict_))

    return out


def get_features_pca(df: pd.DataFrame, num_feat: int, threshold: float=.95):

    return get_orthogonal_features(df, threshold=threshold, num_features=num_feat)





if __name__ == '__main__':

    start = '1990-01-01'
    end = '2022-09-22'

    # features = pd.read_csv('./Data/features_sp.csv', index_col=0)
    #
    # features.index = pd.to_datetime(features.index)
    # features = features.sort_index(ascending=True)
    # for c in features.columns:
    #     tmp = pd.DataFrame(features[c].dropna())
    #     tmp.columns = ['close']
    #     out = plot_min_ffd(tmp)
    #     out.to_csv(f'/home/hjmoon/2022-02논문/results/frac_diff/{c}.csv')

    data = pd.read_csv('./Data/SP500.csv', index_col=0)
    close = data.Close
    high = data.Hihg
    open = data.Open
    low = data.Low
    vol = data.Volume

