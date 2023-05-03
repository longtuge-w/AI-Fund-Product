import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle5
import quantstats as qs
from functools import reduce
from tqdm import trange
from numpy.lib.stride_tricks import as_strided as stride


# read the pkl data
def read_pkl5_data(dir: str):
    f = open(dir, 'rb')
    data = pickle5.load(f)
    f.close()
    return data

# read the member of stocks in the stock index
def read_member(member_dir: str, start_date: int, end_date: int):
    """
    000300.SH: hs300
    000852.SH: zz1000
    000905.SH: zz500
    """
    df = read_pkl5_data(member_dir)
    df.reset_index(inplace=True)
    df.rename(columns={'StkID_str': 'id', 'index': 'date'}, inplace=True)
    df['id'] = df['id'].apply(lambda x: x[:-3])
    df = df.loc[df['id'].apply(lambda x: x.isdigit())]
    df['id'] = df['id'].astype(np.int32)
    df = df.loc[(df['date'] >= start_date) & (df['date'] <= end_date)]
    return df.reset_index(drop=True)

# transform the data in matrix form into three columns dataframe
def stack_pkl_data(dir: str, feature: str):

    df = read_pkl5_data(dir)
    df = df.stack().reset_index()
    df.rename(columns={'level_0': 'date', 'level_1': 'id', 0: feature}, inplace=True)
    df['id'] = df['id'].apply(lambda x: x[:-3])
    df = df.loc[df['id'].apply(lambda x: x.isdigit())]
    df['id'] = df['id'].astype(np.int32)
    return df.reset_index(drop=True)

# get the rolling data of dataframe, it can roll on multiple columns
def roll(df, w, **kwargs):
    v = df.values
    d0, d1 = v.shape
    s0, s1 = v.strides

    a = stride(v, (d0 - (w - 1), w, d1), (s0, s0, s1))

    rolled_df = pd.concat({
        row: pd.DataFrame(values, columns=df.columns)
        for row, values in zip(df.index, a)
    })

    return rolled_df.groupby(level=0, **kwargs)


def get_feature_data(dir: str, name: str):
    df = read_pkl5_data(dir)
    df = df['000985.SH']
    df.name = name
    df = df.to_frame()
    df.reset_index(drop=False, inplace=True)
    df.rename(columns={'index': 'date'}, inplace=True)
    df.dropna(subset=[name], inplace=True)
    return df

# get open, high, low, close data of the index
def get_index_data(open_dir: str, high_dir: str, low_dir: str, close_dir: str):

    df = []
    open_data = get_feature_data(open_dir, 'open')
    high_data = get_feature_data(high_dir, 'high')
    low_data = get_feature_data(low_dir, 'low')
    close_data = get_feature_data(close_dir, 'close')
    df.append(open_data)
    df.append(high_data)
    df.append(low_data)
    df.append(close_data)

    df = reduce(lambda df1, df2: pd.merge(df1, df2, on='date', how='inner'), df)
    return df

# 下跌反弹
def decline_rebound(save_dir: str, df: pd.DataFrame, D, U, start_date: int, end_date: int):

    # create folder to store the unit data
    if not os.path.exists(f'{save_dir}/Signal'):
        os.makedirs(f'{save_dir}/Signal')
    
    df['pre_close'] = df['close'].shift(1)
    df['C1'] = (df['high'] - df['low']) / df['pre_close']
    df['C2'] = (df['high'] - df['pre_close']) / df['pre_close']
    df['C3'] = (df['pre_close'] - df['low']) / df['pre_close']
    df['TR'] = df[['C1', 'C2', 'C3']].abs().max(axis=1)
    df['ATR'] = df['TR'].rolling(60).mean()
    df['return'] = df['close'].pct_change(1)
    df.dropna(subset=['ATR'], inplace=True)
    df['D'], df['U'] = D, U
    df['D'].loc[df['ATR'] > 0.02] = D * np.sqrt(df['ATR'].loc[df['ATR'] > 0.02].values / 0.02)
    df['U'].loc[df['ATR'] > 0.02] = U * np.sqrt(df['ATR'].loc[df['ATR'] > 0.02].values / 0.02)
    df['D'].loc[df['ATR'] < 0.01] = D * np.sqrt(df['ATR'].loc[df['ATR'] < 0.01].values / 0.01)
    df['U'].loc[df['ATR'] < 0.01] = U * np.sqrt(df['ATR'].loc[df['ATR'] < 0.01].values / 0.01)

    def DR(x):
        x.reset_index(drop=True, inplace=True)
        d, u = x['D'].iloc[-1], x['U'].iloc[-1]
        if x['return'].iloc[-1] <= u or x['return'].iloc[-2] > u:
            return 0
        x['ok_point'] = (x['return'] <= u).astype(int)
        x = x.sort_values('date', ascending=False)
        x['ok_point'].iloc[0] = 1
        cumret = ((x['return']+1).cumprod() * x['ok_point'])
        cumret = cumret.replace(0, np.nan)
        cumret = cumret.loc[:cumret[cumret.isnull()].index[0]]
        cumret = cumret.dropna()
        if cumret.max() > 1:
            cumret /= cumret.max()
        if cumret.iloc[-1] < 1-abs(d):
            return 1
        else:
            return 0
    
    df['decline_rebound'] = roll(df, 60).apply(DR)
    df['decline_rebound'] = df['decline_rebound'].shift(60-1)
    df['decline_rebound'] = df['decline_rebound'].replace(0, np.nan)
    df['decline_rebound'] *= df['close']

    df = df.loc[(df['date'] >= start_date) & (df['date'] <= end_date)]
    df = df.loc[:,['date', 'decline_rebound']]
    df.to_csv(f'{save_dir}/Signal/decline_rebound_{start_date}_{end_date}.csv')

    return df

# 三角形收缩突破
def contract_break(save_dir: str, df: pd.DataFrame, C, B, start_date: int, end_date: int):

    # create folder to store the unit data
    if not os.path.exists(f'{save_dir}/Signal'):
        os.makedirs(f'{save_dir}/Signal')

    df['high5d'] = df['high'].rolling(5).max()
    df['low5d'] = df['low'].rolling(5).min()
    df['channel'] = df['high5d'] - df['low5d']
    df['channel_diff'] = df['channel'].diff(1)
    df['return'] = df['close'].pct_change(1)

    def isAllLower(x):
        return (x.abs() <= C).all()

    df['contract'] = 0
    df['contract'].loc[(df['channel_diff'].shift(1) < 0) & 
                       (df['return'].shift(1).rolling(5).apply(isAllLower) > 0)] = 1

    df['contract_break'] = 0
    df['contract_break'].loc[(df['contract'] == 1) & (df['return'] > B)] = 1
    df['contract_break'] = df['contract_break'].replace(0, np.nan)
    df['contract_break'] *= df['close']
    df['contract'] = df['contract'].replace(0, np.nan)
    df['contract'] *= df['close']

    df = df.loc[(df['date'] >= start_date) & (df['date'] <= end_date)]
    df = df.loc[:,['date', 'contract_break']]
    df.to_csv(f'{save_dir}/Signal/contract_break_{start_date}_{end_date}.csv')

    return df

# 顶部切换
def change_top_ind(save_dir: str, df: pd.DataFrame, indDF: pd.DataFrame, citicDF: pd.DataFrame, start_date: int, end_date: int):

    # create folder to store the unit data
    if not os.path.exists(f'{save_dir}/Signal'):
        os.makedirs(f'{save_dir}/Signal')
    
    # calculate ATR60, get the signal when the decline is higher than ATR60
    df.sort_values('date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['pre_close'] = df['close'].shift(1)
    df['C1'] = (df['high'] - df['low']) / df['pre_close']
    df['C2'] = (df['high'] - df['pre_close']) / df['pre_close']
    df['C3'] = (df['pre_close'] - df['low']) / df['pre_close']
    df['TR'] = df[['C1', 'C2', 'C3']].abs().max(axis=1)
    df['ATR60'] = df['TR'].rolling(60).mean()
    df['return'] = df['close'].pct_change(1)
    df['signal_1'] = 0
    df['signal_1'].loc[df['return'] <= -df['ATR60']] = 1

    # calculate the number of industries whose close price reach a maximum within the last 52 weeks
    # get the signal when the number decreases by over 3
    citicDF = citicDF.loc[(citicDF['id'] != 'CI005029.WI') & (citicDF['id'] != 'CI005030.WI')].reset_index(drop=True)
    indDF = pd.merge(indDF, citicDF, on=['id', 'date'], how='right')
    indDF.sort_values(['id', 'date'], inplace=True)
    indDF.reset_index(drop=True, inplace=True)
    indDF['52week'] = indDF.groupby('id')['close'].rolling(250, min_periods=1).max().reset_index(drop=True)
    indDF['cnt'] = 0
    indDF['cnt'].loc[indDF['close'] >= indDF['52week']] = 1
    indSig = indDF.groupby('date')['cnt'].sum().reset_index()
    indSig['diff'] = indSig['cnt'].diff(1)
    indSig['signal_2'] = 0
    indSig['signal_2'].loc[indSig['diff'] <= -3] = 1
    
    df = pd.merge(df, indSig, on='date', how='inner')
    df['change_top_ind'] = 0
    df['change_top_ind'].loc[(df['signal_1'] == 1) & (df['signal_2'] == 1)] = 1
    df['change_top_ind'] = df['change_top_ind'].replace(0, np.nan)
    df = df.loc[(df['date'] >= start_date) & (df['date'] <= end_date)]
    df['change_top_ind'] *= df['close']
    df.to_csv(f'{save_dir}/change_top_ind.csv')
    df = df.loc[:,['date', 'change_top_ind']]
    df.to_csv(f'{save_dir}/Signal/change_top_ind_{start_date}_{end_date}.csv')
    
    return df


# show the distribution of signal
def get_metrics(df: pd.DataFrame, name: str, start_date: int, end_date: int):
    
    df = df.loc[(df['date'] >= start_date) & (df['date'] <= end_date)]
    df['year'] = df['date'].apply(lambda x: int(str(x)[:4]))
    df = df.groupby('year')[name].count().reset_index()
    print(df)

# load CITIC industry data
def get_ind_data(ind_data_dir: str):
    df = pd.read_csv(ind_data_dir)
    df.rename(columns={'S_INFO_WINDCODE': 'id', 'TRADE_DT': 'date', 'S_DQ_PCTCHANGE': 'return', 'S_DQ_CLOSE': 'close',
                       'S_DQ_HIGH': 'high', 'S_DQ_LOW': 'low', 'S_DQ_OPEN': 'open'}, inplace=True)
    idLst = np.sort(pd.unique(df['id']))
    # idLst = idLst[:30]
    df = df[df['id'].isin(idLst)]
    return df.loc[:,['id', 'date', 'return', 'open', 'high', 'low', 'close']]

def get_CITIC_ind(citic_dir: str):
    data = read_pkl5_data(citic_dir)
    data = data.stack()
    data = data.to_frame()
    data.reset_index(drop=False, inplace=True)
    data.rename(columns={'level_0': 'date', 0: 'id'}, inplace=True)
    data = data.loc[:,['date', 'id']]
    data.drop_duplicates(['date', 'id'], inplace=True)
    data = data.loc[data['id'] != 'NoLabel']
    return data

# get momentum factor based on signal "下跌反弹"
def decline_rebound_factor(save_dir: str, indDF: pd.DataFrame, start_date: int, end_date: int, window: int):
    
    df = pd.read_csv(f'{save_dir}/Signal/decline_rebound_{start_date}_{end_date}.csv')
    indDF = indDF.sort_values(['id', 'date']).reset_index(drop=True)
    indDF['factor'] = indDF['return'] - indDF.groupby('id')['return'].rolling(window).mean().reset_index(drop=True)
    indDF[f'return{window}'] = indDF.groupby('id')['close'].shift(-window).reset_index(drop=True) / indDF['close'].reset_index(drop=True) - 1
    df['decline_rebound'] = df['decline_rebound'].fillna(0)
    df['decline_rebound'].loc[df['decline_rebound'] != 0] = 1
    df = pd.merge(df, indDF, on='date')
    df = df.sort_values(['id', 'date']).reset_index(drop=True)
    df = df.loc[:,['id', 'date', 'factor', 'close', f'return{window}', 'decline_rebound']]
    df.to_csv(f'{save_dir}/Signal/decline_rebound_{start_date}_{end_date}_factor.csv')

# get momentum factor based on signal "三角形收缩突破"
def contract_break_factor(save_dir: str, indDF: pd.DataFrame, start_date: int, end_date: int, window: int):

    df = pd.read_csv(f'{save_dir}/Signal/contract_break_{start_date}_{end_date}.csv')
    indDF = indDF.sort_values(['id', 'date']).reset_index(drop=True)
    indDF['factor'] = indDF['return'] / 100
    indDF[f'return{window}'] = indDF.groupby('id')['close'].shift(-window).reset_index(drop=True) / indDF['close'].reset_index(drop=True) - 1
    df['contract_break'] = df['contract_break'].fillna(0)
    df['contract_break'].loc[df['contract_break'] != 0] = 1
    df = pd.merge(df, indDF, on='date')
    df = df.sort_values(['id', 'date']).reset_index(drop=True)
    df = df.loc[:,['id', 'date', 'factor', 'close', f'return{window}', 'contract_break']]
    df.to_csv(f'{save_dir}/Signal/contract_break_{start_date}_{end_date}_factor.csv')

# get momentum factor based on signal "顶部切换"
def change_top_ind_factor(save_dir: str, indDF: pd.DataFrame, start_date: int, end_date: int, window: int):

    df = pd.read_csv(f'{save_dir}/Signal/change_top_ind_{start_date}_{end_date}.csv')
    indDF = indDF.sort_values(['id', 'date']).reset_index(drop=True)
    indDF['factor'] = indDF['return'] / 100
    indDF[f'return{window}'] = indDF.groupby('id')['close'].shift(-window).reset_index(drop=True) / indDF['close'].reset_index(drop=True) - 1
    df['change_top_ind'] = df['change_top_ind'].fillna(0)
    df['change_top_ind'].loc[df['change_top_ind'] != 0] = 1
    df = pd.merge(df, indDF, on='date')
    df = df.sort_values(['id', 'date']).reset_index(drop=True)
    df = df.loc[:,['id', 'date', 'factor', 'close', f'return{window}', 'change_top_ind']]
    df.to_csv(f'{save_dir}/Signal/change_top_ind_{start_date}_{end_date}_factor.csv')

# Do backtesting
def backtest(save_dir: str, trDays_dir: str, start_date: int, end_date: int, name: str, window: int, thres: float, sign: int):

    # create folder to store the unit data
    if not os.path.exists(f'{save_dir}/Report'):
        os.makedirs(f'{save_dir}/Report')

    # load the trading date data within the corresponding time interval
    timeLst = read_pkl5_data(trDays_dir)
    timeLst = timeLst.index.to_numpy()

    df = pd.read_csv(f'{save_dir}/Signal/{name}_{start_date}_{end_date}_factor.csv', index_col=None)
    dates = np.sort(pd.unique(df.loc[df[name] != 0]['date']))

    returnLst, benchLst = [], []
    n_dates = len(dates)
    with trange(n_dates) as date_bar:    
        for i in date_bar:
            date_i = dates[i]
            date_bar.set_description(f'Do backtest on trading day {date_i}')

            if i < n_dates - 1:
                next_date = dates[i+1]
            else:
                next_date = end_date

            df_i = df.loc[df['date'] == date_i]

            if np.where(timeLst == next_date)[0][0] - np.where(timeLst == date_i)[0][0] >= window:
                benchLst.append(df_i[f'return{window}'].mean())
                if sign == 1:
                    df_i = df_i.loc[df_i['factor'] >= df_i['factor'].quantile(1-thres)]
                elif sign == -1:
                    df_i = df_i.loc[df_i['factor'] <= df_i['factor'].quantile(thres)]
                else:
                    raise ValueError(f'The parameter sign should be either 1/-1, get {sign} instead')
                returnLst.append(df_i[f'return{window}'].mean())
            else:
                df_next = df.loc[df['date'] == next_date]
                df_next.rename(columns={'close': 'close_next'}, inplace=True)
                df_i = pd.merge(df_i[['id', 'close', 'factor']], df_next[['id', 'close_next']], on='id')
                df_i['return'] = df_i['close_next'] / df_i['close'] - 1
                benchLst.append(df_i['return'].mean())
                if sign == 1:
                    df_i = df_i.loc[df_i['factor'] >= df_i['factor'].quantile(1-thres)]
                elif sign == -1:
                    df_i = df_i.loc[df_i['factor'] <= df_i['factor'].quantile(thres)]
                else:
                    raise ValueError(f'The parameter sign should be either 1/-1, get {sign} instead')
                returnLst.append(df_i['return'].mean())

    portfolio = pd.DataFrame({'date': dates, 'return': returnLst, 'benchmark': benchLst})
    portfolio['date'] = portfolio['date'].apply(lambda date_i: pd.Timestamp(int(str(date_i)[:4]), int(str(date_i)[4:6]), int(str(date_i)[6:])))
    portfolio.set_index('date', inplace=True)
    print(portfolio)

    # create the report under the path
    report_dir = f'{save_dir}/Report/{name}_{start_date}_{end_date}_{window}_Top{thres}.html'

    qs.reports.html(portfolio['return'], portfolio['benchmark'],
        title=f'Report of long portfolio with factor {name}',
        output=report_dir)
    
# Plot graph to show average monthly return and cumulative return of different portfolios
def plot_graph(save_dir: str, trDays_dir: str, start_date: int, end_date: int, name: str, window: int, q: int):

    # create folder to store the unit data
    if not os.path.exists(f'{save_dir}/Graph'):
        os.makedirs(f'{save_dir}/Graph')

    # load the trading date data within the corresponding time interval
    timeLst = read_pkl5_data(trDays_dir)
    timeLst = timeLst.index.to_numpy()

    df = pd.read_csv(f'{save_dir}/Signal/{name}_{start_date}_{end_date}_factor.csv', index_col=None)
    dates = np.sort(pd.unique(df.loc[df[name] != 0]['date']))

    returnDict = {'date': []}
    qLst = list(range(1,q+1))
    for i in qLst:
        returnDict[i] = []
    n_dates = len(dates)
    with trange(n_dates) as date_bar:    
        for i in date_bar:
            date_i = dates[i]
            returnDict['date'].append(date_i)
            date_bar.set_description(f'Do backtest on trading day {date_i}')

            if i < n_dates - 1:
                next_date = dates[i+1]
            else:
                next_date = end_date

            df_i = df.loc[df['date'] == date_i]

            if np.where(timeLst == next_date)[0][0] - np.where(timeLst == date_i)[0][0] >= window:
                df_i['group'] = pd.qcut(df_i['factor'], q=q, labels=qLst)
                for i in qLst:
                    returnDict[i].append(df_i.loc[df_i['group'] == i][f'return{window}'].mean())
            else:
                df_next = df.loc[df['date'] == next_date]
                df_next.rename(columns={'close': 'close_next'}, inplace=True)
                df_i = pd.merge(df_i[['id', 'close', 'factor']], df_next[['id', 'close_next']], on='id')
                df_i['return'] = df_i['close_next'] / df_i['close'] - 1
                df_i['group'] = pd.qcut(df_i['factor'], q=q, labels=qLst)
                for i in qLst:
                    returnDict[i].append(df_i.loc[df_i['group'] == i]['return'].mean())

    returnDf = pd.DataFrame(returnDict)
    returnDf['date'] = returnDf['date'].apply(lambda date_i: pd.Timestamp(int(str(date_i)[:4]), int(str(date_i)[4:6]), int(str(date_i)[6:])))
    returnDf.set_index('date', inplace=True)
    retDf = returnDf.mean(axis=0)
    returnDf += 1
    returnDf = returnDf.cumprod()
    returnDf.plot()
    plt.title(f'Cumulative Return of Portfolio Based on Signal {name}')
    plt.savefig(f'{save_dir}/Graph/CumReturn_{name}_{start_date}_{end_date}_{window}_{q}.png')
    plt.show()
    plt.close()

    retDf = retDf.to_frame().reset_index(drop=False)
    retDf.rename(columns={'index': 'Portfolio', 0: 'Average return'}, inplace=True)
    retDf['Average return'] = retDf['Average return'] - retDf['Average return'].mean()
    retDf.plot.bar(x='Portfolio', y='Average return')
    plt.title(f'Average Return of Portfolio Based on Signal {name}')
    plt.savefig(f'{save_dir}/Graph/MeanRet_{name}_{start_date}_{end_date}_{window}_{q}.png')
    plt.show()
    plt.close()


# get momentum factor based on signal "下跌反弹"
def decline_rebound_stock_factor(save_dir: str, member_dir: str, close_dir: str, start_date: int, end_date: int, window: int):
    
    df = pd.read_csv(f'{save_dir}/Signal/decline_rebound_{start_date}_{end_date}.csv')
    df.reset_index(drop=True, inplace=True)
    member_df = read_member(member_dir, start_date, end_date)
    close_df = stack_pkl_data(close_dir, 'close')
    stock_df = pd.merge(member_df, close_df, on=['id', 'date'], how='left')
    df = pd.merge(df, stock_df, on='date')
    df = df.sort_values(['id', 'date']).reset_index(drop=True)
    df['return'] = df.groupby('id')['close'].pct_change(1)
    df['factor'] = df['return'] - df.groupby('id')['return'].shift(1).rolling(window).mean()
    df['decline_rebound_factor'] = 0
    df['decline_rebound_factor'].loc[df['decline_rebound'].notnull()] = df['factor'].loc[df['decline_rebound'].notnull()]
    df = df.loc[df['decline_rebound_factor'] != 0]
    df = df.loc[:,['id', 'date', 'decline_rebound_factor']]
    df.to_csv(f'{save_dir}/Signal/decline_rebound_{start_date}_{end_date}_stock_factor.csv')

# get momentum factor based on signal "三角形收缩突破"
def contract_break_stock_factor(save_dir: str, member_dir: str, close_dir: str, start_date: int, end_date: int, window: int):

    df = pd.read_csv(f'{save_dir}/Signal/contract_break_{start_date}_{end_date}.csv', index_col=[0])
    df.reset_index(drop=True, inplace=True)
    member_df = read_member(member_dir, start_date, end_date)
    close_df = stack_pkl_data(close_dir, 'close')
    stock_df = pd.merge(member_df, close_df, on=['id', 'date'], how='left')
    df = pd.merge(df, stock_df, on='date')
    df = df.sort_values(['id', 'date']).reset_index(drop=True)
    df['return'] = df.groupby('id')['close'].pct_change(1)
    df['factor'] = df['return'] - df.groupby('id')['return'].shift(1).rolling(window).mean()
    df['contract_break_factor'] = 0
    df['contract_break_factor'].loc[df['contract_break'].notnull()] = df['factor'].loc[df['contract_break'].notnull()]
    df = df.loc[df['contract_break_factor'] != 0]
    df = df.loc[:,['id', 'date', 'contract_break_factor']]
    df.to_csv(f'{save_dir}/Signal/contract_break_{start_date}_{end_date}_stock_factor.csv')

# get momentum factor based on signal "顶部切换"
def change_top_ind_stock_factor(save_dir: str, member_dir: str, close_dir: str, start_date: int, end_date: int, window: int):

    df = pd.read_csv(f'{save_dir}/Signal/change_top_ind_{start_date}_{end_date}.csv')
    df.reset_index(drop=True, inplace=True)
    member_df = read_member(member_dir, start_date, end_date)
    close_df = stack_pkl_data(close_dir, 'close')
    stock_df = pd.merge(member_df, close_df, on=['id', 'date'], how='left')
    df = pd.merge(df, stock_df, on='date')
    df = df.sort_values(['id', 'date']).reset_index(drop=True)
    df['factor'] = df.groupby('id')['close'].pct_change(1)
    df['change_top_ind_factor'] = 0
    df['change_top_ind_factor'].loc[df['change_top_ind'].notnull()] = df['factor'].loc[df['change_top_ind'].notnull()]
    df = df.loc[df['change_top_ind_factor'] != 0]
    df = df.loc[:,['id', 'date', 'change_top_ind_factor']]
    df.to_csv(f'{save_dir}/Signal/change_top_ind_{start_date}_{end_date}_stock_factor.csv')


def backtest_stock_prep(save_dir: str, trDays_dir: str, start_date: int, end_date: int, nameLst: list):

    # load the trading date data within the corresponding time interval
    timeLst = read_pkl5_data(trDays_dir)
    timeLst = timeLst.index.to_numpy()

    signal_df = []
    for name in nameLst:
        signal_i = pd.read_csv(f'{save_dir}/Signal/{name}_{start_date}_{end_date}_stock_factor.csv', index_col=[0])
        signal_df.append(signal_i)

    signal_df = reduce(lambda df1, df2: pd.merge(df1, df2, on=['id', 'date'], how='outer'), signal_df)
    f_cols = [f'{name}_factor' for name in nameLst]
    signal_df[f_cols] = signal_df[f_cols].fillna(0)
    signal_df['factor'] = signal_df[f_cols].sum(axis=1)
    signal_df.sort_values(by=['id', 'date'], inplace=True)
    dateLst = np.sort(pd.unique(signal_df['date']))
    signal = pd.DataFrame()

    # calculate factor values by dates
    for date_i in dateLst:
        signal_i = signal_df.loc[signal_df['date'] == date_i].copy()
        ind = np.where(timeLst == date_i)[0][0]
        temp_time = timeLst[ind-10:ind]
        last_time = np.intersect1d(dateLst, temp_time)
        # if there are no other signal within the last 10 dates, we just use the factor on current day,
        # otherwise we add the effect of the previous signals
        if len(last_time) == 0:
            signal_i = signal_i.loc[:,['id', 'date', 'factor']]
            signal = signal.append(signal_i)
            continue
        last_signal = signal_df[signal_df['date'].isin(last_time)].copy()
        last_signal = last_signal.loc[:,['id', 'date', 'factor']]
        last_signal.rename(columns={'factor': 'last'}, inplace=True)
        for l in last_time:
            last_ind = np.where(timeLst == l)[0][0]
            last_signal_i = last_signal.loc[last_signal['date'] == l].copy()
            signal_i = pd.merge(signal_i, last_signal_i, on=['id', 'date'])
            signal_i['factor'] += signal_i['last'] * (2 ** ((last_ind - ind) / 10))
            del signal_i['last']
        signal_i = signal_i.loc[:,['id', 'date', 'factor']]
        signal = signal.append(signal_i)

    signal.reset_index(drop=True, inplace=True)
    new_name = '_'.join(nameLst)

    signal.to_csv(f'{save_dir}/Signal/{new_name}_{start_date}_{end_date}_stock_factor.csv')

# Do backtesting
def backtest_stock(save_dir: str, trDays_dir: str, close_dir: str, start_date: int, end_date: int, nameLst: list, thres: int):

    # create folder to store the unit data
    if not os.path.exists(f'{save_dir}/Report'):
        os.makedirs(f'{save_dir}/Report')

    if not os.path.exists(f'{save_dir}/Graph'):
        os.makedirs(f'{save_dir}/Graph')

    # load the trading date data within the corresponding time interval
    timeLst = read_pkl5_data(trDays_dir)
    timeLst = timeLst.index.to_numpy()

    new_name = '_'.join(nameLst)
    signal = pd.read_csv(f'{save_dir}/Signal/{new_name}_{start_date}_{end_date}_stock_factor.csv', index_col=0)
    close_df = stack_pkl_data(close_dir, 'close')
    dateLst = np.sort(pd.unique(signal['date']))

    # calculate returns for long portfolio, short portfolio, and benchmark
    longLst, shortLst, benchLst = [], [], []

    with trange(len(dateLst)) as date_bar:
        for i in date_bar:
            date_i = dateLst[i]
            date_bar.set_description(f'Doing backtesting on trading date {date_i}')
            ind = np.where(timeLst == date_i)[0][0]
            temp_time = timeLst[ind+1:ind+20+1]
            last_time = np.intersect1d(dateLst, temp_time)
            # if there are another signal within next 20 dates, we change our factor into the ones on that day,
            # otherwise we hold the stock for 20 dates
            if len(last_time) == 0:
                next_date = timeLst[ind+20]
            else:
                next_date = np.min(last_time)
            # calculate the returns for different portfolios
            next_i = close_df.loc[close_df['date'] == next_date].copy()
            next_i = next_i.loc[:,['id', 'close']]
            next_i.rename(columns={'close': 'next'}, inplace=True)
            close_i = close_df.loc[close_df['date'] == date_i].copy()
            close_i = close_i.loc[:,['id', 'close']]
            signal_i = signal.loc[signal['date'] == date_i].copy()
            signal_i = pd.merge(signal_i, next_i, on='id')
            signal_i = pd.merge(signal_i, close_i, on='id')
            signal_i['return'] = signal_i['next'] / signal_i['close'] - 1
            bench_i = signal_i['return'].mean()
            benchLst.append(bench_i)
            signal_i.sort_values(by='factor', inplace=True)
            long_i = signal_i.iloc[-thres:]['return'].mean()
            longLst.append(long_i)
            short_i = signal_i.iloc[:thres]['return'].mean()
            shortLst.append(short_i)

    portfolio = pd.DataFrame({'date': dateLst, 'return': longLst, 'benchmark': benchLst})
    portfolio['date'] = portfolio['date'].astype(str).apply(lambda date_i: pd.Timestamp(int(date_i[:4]), int(date_i[4:6]), int(date_i[6:])))
    portfolio.set_index(keys='date', inplace=True)
    print(portfolio)

    # create the report under the path
    report_dir = f'{save_dir}/Report/{new_name}_{start_date}_{end_date}_Top{thres}.html'

    qs.reports.html(portfolio['return'], portfolio['benchmark'],
        title=f'Report of long portfolio with factor {new_name}',
        output=report_dir)

    # plot the cumulative return of long, short, and long-short portfolios
    longLst, shortLst = np.array(longLst), np.array(shortLst)
    long_short = longLst - shortLst
    longLst += 1
    shortLst += 1
    long_short += 1
    cum_long = np.cumprod(longLst)
    cum_short = np.cumprod(shortLst)
    cum_long_short = np.cumprod(long_short)

    _, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    strategy = pd.DataFrame({'date': dateLst, 'long': cum_long, 'short': cum_short, 'long-short': cum_long_short})
    strategy['date'] = strategy['date'].astype(str).apply(lambda date_i: pd.Timestamp(int(date_i[:4]), int(date_i[4:6]), int(date_i[6:])))
    strategy.set_index(keys='date', inplace=True)
    strategy[['long', 'short']].plot(ax=ax1, marker='o')
    ax2.plot(strategy.index, strategy['long-short'], color='black', marker='o', label='long-short')
    plt.title('Cumulative return')
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    lines = lines_1 + lines_2
    labels = labels_1 + labels_2
    ax1.legend(lines, labels, loc='upper left')
    plt.savefig(f'{save_dir}/Graph/CumReturn_{new_name}_{start_date}_{end_date}.png')
    plt.show()
    plt.close()