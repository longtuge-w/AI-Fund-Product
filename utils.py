import os
import sys
import pickle
import numpy as np
import pandas as pd
from tqdm import trange

import warnings
warnings.filterwarnings("ignore")


# read the pkl data
def read_pkl(dir: str):
    f = open(dir,'rb')
    data = pickle.load(f)
    f.close()
    return data

# write the pkl data
def write_pkl(dir: str, data):
    with open(dir, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    handle.close()


def RPS(df: pd.DataFrame, window: int=20):
    df['ret'] = df['ret'] / 100 + 1
    df.set_index(keys='date', inplace=True)
    df = (df.groupby(by='id').rolling(window=window)['ret'].apply(np.prod) - 1).reset_index(drop=False)
    df.dropna(subset=['ret'], inplace=True)
    df['RPS_%s' % window] = df.groupby(by='date')['ret'].rank(pct=True) * 100
    return df


def signal(df: pd.DataFrame, window: int=60):
    # The difference between upwards and downwards volatility
    df['vol_diff'] = ((df['high'] - df['open']) - (df['open'] - df['low'])) / df['open']
    # if window is an integer, we calculate the rolling average with the constant window
    if isinstance(window, int):
        df.set_index(keys='date', inplace=True)
        df = df.groupby(by='id').rolling(window=window)['vol_diff'].apply(np.mean).reset_index(drop=False)
        df.rename(columns={'vol_diff': 'MA_vol_diff'}, inplace=True)
    # if the window is a list, we choose the window for rolling average based on the RPS
    # higher RPS corresponds to larger window
    elif isinstance(window, list):
        num_group = len(window)
        df.set_index(keys='date', inplace=True)
        for i in range(num_group):
            low_RPS = i * 100 / num_group
            high_RPS = (i+1) * 100 / num_group
            df['window'] = np.nan
            df['window'].loc[(df['RPS'] > low_RPS) & (df['RPS'] <= high_RPS)] = window[i]
        df['window'] = df['window'].fillna(window[0])

        def cat_rolling(x):
            x.reset_index(drop=False, inplace=True)
            x['MA_vol_diff'] = None
            for idx, row in x.iterrows():
                if idx + 1 >= row['window']:
                    temp = x.iloc[int(idx-row['window']+1):int(idx+1),]['vol_diff']
                    x.loc[idx, 'MA_vol_diff'] = temp.mean()
            x.set_index('date', inplace=True)
            return x['MA_vol_diff']

        df = df.groupby(by='id').apply(cat_rolling).reset_index(drop=False)

    df.dropna(subset=['MA_vol_diff'], inplace=True)
    df['signal'] = 1
    df['signal'].loc[df['MA_vol_diff'] <= 0] = 0
    print(df)

    return df[['id', 'date', 'signal']].reset_index(drop=True)


# get and store the all the factor data
def get_RPS(save_dir: str, stock: pd.DataFrame, window: int):
    # check if the folders for factor data are existed
    if not os.path.exists('%s/Factors/RPS_%s' % (save_dir, window)):
        os.makedirs('%s/Factors/RPS_%s' % (save_dir, window))
    # get daily high frequency data
    stock.sort_values(by=['id', 'date'], inplace=True)
    stock = RPS(stock, window)
    dateLst = pd.unique(stock['date'])
    with trange(len(dateLst)) as date_bar:    
        for i in date_bar:
            date_i = dateLst[i]
            date_bar.set_description('Preprocessing data on date %s' % date_i)
            # get the factor data when needed
            if os.path.exists('%s/Factors/RPS_%s/%s.csv' % (save_dir, window, date_i)):
                continue
            df_i = stock.loc[stock['date'] == date_i]
            df_i.to_csv('%s/Factors/RPS_%s/%s.csv' % (save_dir, window, date_i))

# get and store the all the factor data
def get_signal(save_dir: str, stock: pd.DataFrame, start_date: int, end_date: int, window=[20,40,60,80,100]):
    # check if the folders for factor data are existed
    if not os.path.exists('%s/Signal/RPS' % (save_dir)):
        os.makedirs('%s/Signal/RPS' % (save_dir))
    # get daily high frequency data
    dirLst = os.listdir('%s/Factors/RPS' % (save_dir))
    dirLst = sorted([dir for dir in dirLst if int(dir[:8]) >= start_date and int(dir[:8]) <= end_date])
    df = pd.DataFrame()
    with trange(len(dirLst)) as dir_bar:    
        for i in dir_bar:
            dir_i = dirLst[i]
            dir_bar.set_description('Preprocessing file %s' % dir_i)
            # get the factor data when needed
            df_i = pd.read_csv('%s/Factors/RPS/%s' % (save_dir, dir_i), index_col=[0])
            df = df.append(df_i)
    df.reset_index(drop=True, inplace=True)
    df.rename(columns={'ret': 'rolling_ret'}, inplace=True)
    df = pd.merge(df, stock, on=['id', 'date'], how='left')
    # get the signal for the month line stock data
    df = signal(df, window)
    df.to_csv('%s/Signal/RPS_signal_%s_%s.csv' % (save_dir, start_date, end_date))

# read the stock data
def read_stock_data(stocklist_dir: str, save_dir: str, start_date: int, end_date: int, features: list, benchmark: str='hs300'):
    # create folders if needed
    if not os.path.exists('%s/Stock_Data' % save_dir):
        os.makedirs('%s/Stock_Data' % save_dir)
    
    if os.path.exists('%s/Stock_Data/Stock_Data_%s_%s.pkl' % (save_dir, start_date, end_date)):
        print('Required stock data already exists')
        return read_pkl('%s/Stock_Data/Stock_Data_%s_%s.pkl' % (save_dir, start_date, end_date))

    # read the stock data for later use
    f = open(stocklist_dir,'rb')
    stockLst = pickle.load(f)
    StockData = pd.DataFrame()

    with trange(len(stockLst)) as stock_bar:    
        for i in stock_bar:
            stockDF = stockLst[i]
            stock_id = stockDF['id'].iloc[0]
            stock_bar.set_description('Processing stock number %s'%(stock_id))
            # We only need the stock data within the backtest period
            stockDF = stockDF.loc[(stockDF['tdate'] >= start_date) & (stockDF['tdate'] <= end_date)].copy()
            # choose stock pool
            if benchmark == 'hs300':
                stockDF = stockDF.loc[stockDF['member'] == 1]
            elif benchmark == 'zz500':
                stockDF = stockDF.loc[stockDF['member'] == 2]
            elif benchmark == 'zz1000':
                stockDF = stockDF.loc[stockDF['member'] == 3]
            elif benchmark == 'othershare':
                stockDF = stockDF.loc[stockDF['member'] == 4]
            elif benchmark == 'allAshare':
                pass
            else:
                raise ValueError("The parameter benchmark should be hs300/zz500/zz1000/othershare/allAshare, get %s instead" % benchmark)

            if stockDF.empty:
                continue

            # if the stock satisfies all the requirements, we add it to the stock pool
            stockDF.rename(columns={'tdate': 'date'}, inplace=True)
            stockDF = stockDF.loc[:,features]
            StockData = StockData.append(stockDF)

    StockData.reset_index(inplace=True, drop=True)
    write_pkl('%s/Stock_Data/Stock_Data_%s_%s.pkl' % (save_dir, start_date, end_date), StockData)
    return StockData