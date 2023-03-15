import os
import sys
import math
import copy
import numpy as np
import pandas as pd
import pickle
import pickle5
import joblib
import statsmodels.api as sm
import quantstats as qs
import scipy.stats as sp
import torch as th
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
# from sklearn.preprocessing import MinMaxScaler
from functools import reduce
from tqdm import trange
# from lightgbm import LGBMRegressor
from LSTM import LSTMModel
from GRU import GRUModel
from ALSTM import ALSTMModel
from TCN import TCNModel
from Transformer import Transformer
from LSTM_Rank import LSTMRankModel
from GRU_Rank import GRURankModel
from ALSTM_Rank import ALSTMRankModel
from TCN_Rank import TCNRankModel
from Transformer_Rank import TransformerRank


# th.backends.cudnn.enabled = False


if th.cuda.is_available():
    device = th.device('cuda')
else:
    device = th.device('cpu')


# Variables
BATCH_SIZE = 1024
FACTOR_BATCH_SIZE = 32
LEARNING_RATE = 1e-4
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-5
MAX_EPOCH = 1
RANDOM_STATE = 1
DEFAULT_EPS = 1e-10
PADDED_Y_VALUE = -1
TAU = 0.5


class Data(Dataset):
    """
    The simple Dataset object from th that can produce batchs of data
    """
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

# read the pkl data
def read_pkl_data(dir: str):
    f = open(dir, 'rb')
    data = pickle.load(f)
    f.close()
    return data

# write the pkl data
def write_pkl_data(dir: str, data):
    with open(dir, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    handle.close()

# read the pkl data
def read_pkl5_data(dir: str):
    f = open(dir, 'rb')
    data = pickle5.load(f)
    f.close()
    return data

# write the pkl data
def write_pkl5_data(dir: str, data):
    with open(dir, 'wb') as handle:
        pickle5.dump(data, handle, protocol=pickle5.HIGHEST_PROTOCOL)
    handle.close()

# read the data of all the trading dates
def read_tradeDate_data(time_dir: str, start_date: int, end_date: int):
    print('loading trading dates data ...')
    timeLst = read_pkl_data(time_dir)
    timeLst = timeLst[(timeLst >= start_date) & (timeLst <= end_date)]
    return timeLst

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
    return df

# read the stock data
def read_stock_data(stocklist_dir: str, save_dir: str, start_date: int, end_date: int, features: list, benchmark: str='allAshare'):

    if not os.path.exists(f'{save_dir}/Data/Stock/{benchmark}'):
        os.makedirs(f'{save_dir}/Data/Stock/{benchmark}')

    if os.listdir(f'{save_dir}/Data/Stock/{benchmark}'):
        exist_date = max(os.listdir(f'{save_dir}/Data/Stock/{benchmark}'))
        exist_date = int(exist_date[:8])
        start_date = max(exist_date, start_date)

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
            stockDF = stockDF.loc[(stockDF['tdate'] >= start_date) & (stockDF['tdate'] <= end_date)]
            if benchmark != 'allAshare':
                # only choose stocks in a certain stock index
                if benchmark == 'hs300':
                    stockDF = stockDF.loc[stockDF['member'] == 1]
                elif benchmark == 'zz500':
                    stockDF = stockDF.loc[stockDF['member'] == 2]
                elif benchmark == 'zz800':
                    stockDF = stockDF.loc[(stockDF['member'] == 1) | (stockDF['member'] == 2)]
                elif benchmark == 'othershare':
                    stockDF = stockDF.loc[stockDF['member'] == 4]
                elif benchmark == 'Top1800':
                    stockDF = stockDF.loc[(stockDF['member'] == 1) | (stockDF['member'] == 2) | (stockDF['member'] == 3)]

            if stockDF.empty:
                continue

            # if the stock satisfies all the requirements, we add it to the stock pool
            stockDF.rename(columns={'tdate': 'date'}, inplace=True)
            # calculate the feature vwap = amount/volume
            stockDF['vwap'] = stockDF['amount'] / stockDF['volume']
            stockDF = stockDF[['date', 'id', 'vwap'] + features]

            if not stockDF.empty:
                StockData = StockData.append(stockDF)

    StockData.reset_index(inplace=True, drop=True)
    # save the stock data by dates
    dateLst = np.sort(pd.unique(StockData['date']))
    with trange(len(dateLst)) as date_bar:    
        for i in date_bar:
            date_i = dateLst[i]
            date_bar.set_description(f'Saving stock data on trading date {date_i}')
            stock_i = StockData.loc[StockData['date'] == date_i]
            stock_i.to_csv(f'{save_dir}/Data/Stock/{benchmark}/{date_i}.csv', index=False)


def stack_pkl_data(dir: str, start_date: int, feature: str):

    df = read_pkl5_data(dir)
    df = df.stack().reset_index()
    df.rename(columns={'level_0': 'date', 'level_1': 'id', 0: feature}, inplace=True)
    df = df.loc[df['date'] >= start_date]
    df['id'] = df['id'].apply(lambda x: x[:-3])
    df = df.loc[df['id'].apply(lambda x: x.isdigit())]
    df['id'] = df['id'].astype(np.int32)
    return df


def read_stock_data_new(save_dir: str, open_dir: str, high_dir: str, low_dir: str, close_dir: str, volume_dir: str, 
                        vwap_dir: str, start_date: int, end_date: int):

    if not os.path.exists(f'{save_dir}/Data/Stock/allAshare'):
        os.makedirs(f'{save_dir}/Data/Stock/allAshare')

    if os.listdir(f'{save_dir}/Data/Stock/allAshare'):
        exist_date = max(os.listdir(f'{save_dir}/Data/Stock/allAshare'))
        exist_date = int(exist_date[:8])
        start_date = max(exist_date, start_date)

    open_df = stack_pkl_data(open_dir, start_date, 'open')
    close_df = stack_pkl_data(close_dir, start_date, 'close')
    high_df = stack_pkl_data(high_dir, start_date, 'high')
    low_df = stack_pkl_data(low_dir, start_date, 'low')
    volume_df = stack_pkl_data(volume_dir, start_date, 'volume')
    vwap_df = stack_pkl_data(vwap_dir, start_date, 'vwap')

    dfLst = [open_df, close_df, high_df, low_df, volume_df, vwap_df]
    StockData = reduce(lambda df1, df2: pd.merge(df1, df2, on=['id', 'date'], how='inner'), dfLst)
    StockData = StockData.loc[(StockData['date'] >= start_date) & (StockData['date'] <= end_date)]
    StockData = StockData[['date', 'id', 'vwap', 'high', 'open', 'low', 'close', 'volume']]

    # save the stock data by dates
    dateLst = np.sort(pd.unique(StockData['date']))
    with trange(len(dateLst)) as date_bar:    
        for i in date_bar:
            date_i = dateLst[i]
            date_bar.set_description(f'Saving stock data on trading date {date_i}')
            stock_i = StockData.loc[StockData['date'] == date_i]
            stock_i.to_csv(f'{save_dir}/Data/Stock/allAshare/{date_i}.csv', index=False)


# read and store the return data of stocks
def read_return_data(save_dir: str, return_dir: str, member_dir: str, start_date: int, end_date: int, bench: str, T: int):
    # create folder to store the return data
    if not os.path.exists(f'{save_dir}/Data/Return/allAshare/{T}'):
        os.makedirs(f'{save_dir}/Data/Return/allAshare/{T}')

    if os.listdir(f'{save_dir}/Data/Return/allAshare/{T}'):
        exist_date = max(os.listdir(f'{save_dir}/Data/Return/allAshare/{T}'))
        exist_date = int(exist_date[:8])
        start_date = max(exist_date, start_date)

    # read the matrix of stock return
    if return_dir.split('.')[-1] == 'pkl':
        ret = read_pkl5_data(return_dir)
    elif return_dir.split('.')[-1] == 'h5':
        ret = pd.read_hdf(return_dir)

    ret = ret.stack().reset_index()
    ret.rename(columns={'level_0': 'date', 'level_1': 'id', 0: 'return'}, inplace=True)

    if ret['id'].dtype == object:
        ret['id'] = ret['id'].apply(lambda x: x[:-3])
        ret = ret.loc[ret['id'].apply(lambda x: x.isdigit())]
        ret['id'] = ret['id'].astype(np.int32)

    # get the member of stock index
    if bench != 'allAshare':
        member = read_member(member_dir, start_date, end_date)
        df = pd.merge(member, ret, on=['id', 'date'])
    else:
        df = ret

    # save the return data by dates
    dateLst = np.sort(pd.unique(df['date']))
    with trange(len(dateLst)) as date_bar:    
        for i in date_bar:
            date_i = dateLst[i]
            date_bar.set_description(f'Saving return data on trading date {date_i}')
            df_i = df.loc[df['date'] == date_i]
            df_i.to_csv(f'{save_dir}/Data/Return/allAshare/{T}/{date_i}.csv', index=False)

# read the stock data
def derive_return_data(stocklist_dir: str, save_dir: str, start_date: int, end_date: int, T: int, gap: bool):

    # create the folder if exists
    if gap:
        if not os.path.exists(f'{save_dir}/Data/Return/allAshare/T+1_T+{T+1}'):
            os.makedirs(f'{save_dir}/Data/Return/allAshare/T+1_T+{T+1}')
        if os.listdir(f'{save_dir}/Data/Return/allAshare/T+1_T+{T+1}'):
            exist_date = max(os.listdir(f'{save_dir}/Data/Return/allAshare/T+1_T+{T+1}'))
            exist_date = int(exist_date[:8])
            start_date = max(exist_date, start_date)
    else:
        if not os.path.exists(f'{save_dir}/Data/Return/allAshare/T_T+{T}'):
            os.makedirs(f'{save_dir}/Data/Return/allAshare/T_T+{T}')
        if os.listdir(f'{save_dir}/Data/Return/allAshare/T_T+{T}'):
            exist_date = max(os.listdir(f'{save_dir}/Data/Return/allAshare/T_T+{T}'))
            exist_date = int(exist_date[:8])
            start_date = max(exist_date, start_date)

    # read the stock data for later use
    f = open(stocklist_dir,'rb')
    stockLst = pickle.load(f)
    StockData = pd.DataFrame()

    # save the return data by trading dates
    with trange(len(stockLst)) as stock_bar:    
        for i in stock_bar:
            stockDF = stockLst[i]
            stock_id = stockDF['id'].iloc[0]
            stock_bar.set_description('Processing stock number %s'%(stock_id))

            # if the stock satisfies all the requirements, we add it to the stock pool
            stockDF.rename(columns={'tdate': 'date'}, inplace=True)
            if gap:
                stockDF['return'] = stockDF['close'].shift(-(T+1)) / stockDF['close'].shift(-1) - 1
            else:
                stockDF['return'] = stockDF['close'].shift(-T) / stockDF['close'] - 1
            stockDF = stockDF.loc[(stockDF['date'] >= start_date) & (stockDF['date'] <= end_date)]
            stockDF = stockDF.loc[:,['id', 'date', 'return']]
            stockDF = stockDF.dropna(subset=['return'])
            stockDF.sort_values(by=['id', 'date'], inplace=True)

            if not stockDF.empty:
                StockData = StockData.append(stockDF)

    # save the stock data by dates
    dateLst = np.sort(pd.unique(StockData['date']))
    # save return data by trading dates
    with trange(len(dateLst)) as date_bar:
        for i in date_bar:
            date_i = dateLst[i]
            date_bar.set_description(f'Saving return data on trading date {date_i}')
            stock_i = StockData.loc[StockData['date'] == date_i]
            if gap:
                stock_i.to_csv(f'{save_dir}/Data/Return/allAshare/T+1_T+{T+1}/{date_i}.csv', index=False)
            else:
                stock_i.to_csv(f'{save_dir}/Data/Return/allAshare/T_T+{T}/{date_i}.csv', index=False)


# read the stock data
def derive_return_data_close_new(close_dir: str, save_dir: str, start_date: int, end_date: int, T: int, gap: bool):

    # create the folder if exists
    if gap:
        if not os.path.exists(f'{save_dir}/Data/Return/allAshare/T+1_T+{T+1}'):
            os.makedirs(f'{save_dir}/Data/Return/allAshare/T+1_T+{T+1}')
        if os.listdir(f'{save_dir}/Data/Return/allAshare/T+1_T+{T+1}'):
            exist_date = max(os.listdir(f'{save_dir}/Data/Return/allAshare/T+1_T+{T+1}'))
            exist_date = int(exist_date[:8])
            start_date = max(exist_date, start_date)
    else:
        if not os.path.exists(f'{save_dir}/Data/Return/allAshare/T_T+{T}'):
            os.makedirs(f'{save_dir}/Data/Return/allAshare/T_T+{T}')
        if os.listdir(f'{save_dir}/Data/Return/allAshare/T_T+{T}'):
            exist_date = max(os.listdir(f'{save_dir}/Data/Return/allAshare/T_T+{T}'))
            exist_date = int(exist_date[:8])
            start_date = max(exist_date, start_date)

    stockDF = stack_pkl_data(close_dir, start_date, 'close')
    stockDF.sort_values(['id', 'date'], inplace=True)

    if gap:
        stockDF['return'] = stockDF.groupby('id')['close'].shift(-(T+1)) / stockDF.groupby('id')['close'].shift(-1) - 1
    else:
        stockDF['return'] = stockDF.groupby('id')['close'].shift(-T) / stockDF.groupby('id')['close'] - 1

    stockDF = stockDF.loc[(stockDF['date'] >= start_date) & (stockDF['date'] <= end_date)]
    stockDF = stockDF.loc[:,['id', 'date', 'return']]
    stockDF = stockDF.dropna(subset=['return'])
    stockDF.sort_values(by=['id', 'date'], inplace=True)

    # save the stock data by dates
    dateLst = np.sort(pd.unique(stockDF['date']))
    # save return data by trading dates
    with trange(len(dateLst)) as date_bar:
        for i in date_bar:
            date_i = dateLst[i]
            date_bar.set_description(f'Saving return data on trading date {date_i}')
            stock_i = stockDF.loc[stockDF['date'] == date_i]
            if gap:
                stock_i.to_csv(f'{save_dir}/Data/Return/allAshare/T+1_T+{T+1}/{date_i}.csv', index=False)
            else:
                stock_i.to_csv(f'{save_dir}/Data/Return/allAshare/T_T+{T}/{date_i}.csv', index=False)


# read the stock data
def derive_return_data_vwap_new(vwap_dir: str, save_dir: str, start_date: int, end_date: int, T: int, gap: bool):

    # create the folder if exists
    if gap:
        if not os.path.exists(f'{save_dir}/Data/Return/allAshare/T+1_T+{T+1}'):
            os.makedirs(f'{save_dir}/Data/Return/allAshare/T+1_T+{T+1}')
        if os.listdir(f'{save_dir}/Data/Return/allAshare/T+1_T+{T+1}'):
            exist_date = max(os.listdir(f'{save_dir}/Data/Return/allAshare/T+1_T+{T+1}'))
            exist_date = int(exist_date[:8])
            start_date = max(exist_date, start_date)
    else:
        if not os.path.exists(f'{save_dir}/Data/Return/allAshare/T_T+{T}'):
            os.makedirs(f'{save_dir}/Data/Return/allAshare/T_T+{T}')
        if os.listdir(f'{save_dir}/Data/Return/allAshare/T_T+{T}'):
            exist_date = max(os.listdir(f'{save_dir}/Data/Return/allAshare/T_T+{T}'))
            exist_date = int(exist_date[:8])
            start_date = max(exist_date, start_date)

    stockDF = stack_pkl_data(vwap_dir, start_date, 'close')
    stockDF.sort_values(['id', 'date'], inplace=True)

    if gap:
        stockDF['return'] = stockDF.groupby('id')['vwap'].shift(-(T+1)) / stockDF.groupby('id')['vwap'].shift(-1) - 1
    else:
        stockDF['return'] = stockDF.groupby('id')['vwap'].shift(-T) / stockDF.groupby('id')['vwap'] - 1

    stockDF = stockDF.loc[(stockDF['date'] >= start_date) & (stockDF['date'] <= end_date)]
    stockDF = stockDF.loc[:,['id', 'date', 'return']]
    stockDF = stockDF.dropna(subset=['return'])
    stockDF.sort_values(by=['id', 'date'], inplace=True)

    # save the stock data by dates
    dateLst = np.sort(pd.unique(stockDF['date']))
    # save return data by trading dates
    with trange(len(dateLst)) as date_bar:
        for i in date_bar:
            date_i = dateLst[i]
            date_bar.set_description(f'Saving return data on trading date {date_i}')
            stock_i = stockDF.loc[stockDF['date'] == date_i]
            if gap:
                stock_i.to_csv(f'{save_dir}/Data/Return/allAshare/T+1_T+{T+1}/{date_i}.csv', index=False)
            else:
                stock_i.to_csv(f'{save_dir}/Data/Return/allAshare/T_T+{T}/{date_i}.csv', index=False)

# read the factor data stored as csv files
def read_factor_data_csv(save_dir: str, factor_dir: str, member_dir: str, start_date: int, end_date: int, name: str, bench: str):
    # get the list of factors
    factorLst = os.listdir(factor_dir)

    for c in factorLst:
        # create folder to store factor data
        if not os.path.exists(f'{save_dir}/Data/{name}/{c}/{bench}'):
            os.makedirs(f'{save_dir}/Data/{name}/{c}/{bench}')
        if os.listdir(f'{save_dir}/Data/{name}/{c}/{bench}'):
            exist_date = max(os.listdir(f'{save_dir}/Data/{name}/{c}/{bench}'))
            exist_date = int(exist_date[:8])
            start_date = max(exist_date, start_date)
        # get the member of stock index
        if bench != 'allAshare':
            member = read_member(member_dir, start_date, end_date)
        # concatenate all the required daily data
        dirLst = os.listdir(f'{factor_dir}/{c}/data')
        dirLst = [i for i in dirLst if int(i[:-4]) >= start_date and int(i[:-4]) <= end_date]
        # read and stack all the factor data
        data_i = pd.DataFrame()
        with trange(len(dirLst)) as dir_bar:    
            for i in dir_bar:
                dir = dirLst[i]
                date_i = int(dir[:-4])
                dir_bar.set_description(f'Loading {c} data on date {date_i}')
                df_i = pd.read_csv(f'{factor_dir}/{c}/data/{dir}')
                df_i.rename(columns={'factor_value': c}, inplace=True)
                df_i['date'] = date_i
                df_i['stock'] = df_i['stock'].apply(lambda x: x[:-3])
                df_i = df_i.loc[df_i['stock'].apply(lambda x: x.isdigit())]
                df_i['stock'] = df_i['stock'].astype(np.int32)
                df_i.rename(columns={'stock': 'id'}, inplace=True)
                df_i.dropna(subset=[c], inplace=True)
                data_i = data_i.append(df_i)
        # get the stocks from the index
        if bench != 'allAshare':
            df = pd.merge(member, data_i, on=['id', 'date'])
        else:
            df = data_i
        # store the factor data by dates
        dateLst = np.sort(pd.unique(df['date']))
        with trange(len(dateLst)) as date_bar:    
            for i in date_bar:
                date_i = dateLst[i]
                date_bar.set_description(f'Saving factor data {c} from {name} on trading date {date_i}')
                df_i = df.loc[df['date'] == date_i]
                df_i.to_csv(f'{save_dir}/Data/{name}/{c}/{bench}/{date_i}.csv', index=False)

# read the factor data stored as pkl files
def read_factor_data_pkl(save_dir: str, factor_dir: str, member_dir: str, start_date: int, end_date: int, name: str, bench: str):
    # get the list of factors
    dirLst = os.listdir(factor_dir)

    for dir in dirLst:
        df_i = read_pkl5_data(f'{factor_dir}/{dir}')
        feature_i = dir.split('_')[0]
        # create folder to store the factor data
        if not os.path.exists(f'{save_dir}/Data/{name}/{feature_i}/{bench}'):
            os.makedirs(f'{save_dir}/Data/{name}/{feature_i}/{bench}')
        if os.listdir(f'{save_dir}/Data/{name}/{feature_i}/{bench}'):
            exist_date = max(os.listdir(f'{save_dir}/Data/{name}/{feature_i}/{bench}'))
            exist_date = int(exist_date[:8])
            start_date = max(exist_date, start_date)
        df_i = df_i.stack().reset_index()
        df_i.rename(columns={'level_0': 'date', 'level_1': 'id', 0: feature_i}, inplace=True)
        df_i = df_i.loc[df_i['date'] >= start_date]
        df_i['id'] = df_i['id'].apply(lambda x: x[:-3])
        df_i = df_i.loc[df_i['id'].apply(lambda x: x.isdigit())]
        df_i['id'] = df_i['id'].astype(np.int32)
        # get the member of stock index
        if bench != 'allAshare':
            member = read_member(member_dir, start_date, end_date)
            # select the stocks in the stock index
            df = pd.merge(member, df_i, on=['id', 'date'])
        else:
            df = df_i.copy()
        # store the stock data by dates
        dateLst = np.sort(pd.unique(df['date']))
        with trange(len(dateLst)) as date_bar:    
            for i in date_bar:
                date_i = dateLst[i]
                date_bar.set_description(f'Saving factor data {feature_i} from {name} on trading date {date_i}')
                df_i = df.loc[df['date'] == date_i]
                df_i.to_csv(f'{save_dir}/Data/{name}/{feature_i}/{bench}/{date_i}.csv', index=False)

# pearson correlation
def pearson_corr(x, y):
    vx = x - x.mean()
    vy = y - y.mean()
    cost = (vx * vy).sum() / ((vx ** 2).sum().sqrt() * (vy ** 2).sum().sqrt())
    return cost

# this loss is the sum of negative value of IC over each stock
class IC_loss(nn.Module):
    def __init__(self):
        super(IC_loss, self).__init__()

    def forward(self, logits, target):
        return -pearson_corr(logits, target)

# this loss is basically the IC loss with penalty on correlation between different factors
class IC_loss_penalty(nn.Module):
    def __init__(self, phi=1e-4):
        super(IC_loss_penalty, self).__init__()
        self.phi = phi

    def forward(self, logits, target):
        c = logits.mean(dim=1)
        cov = th.corrcoef(logits.permute(1,0))
        penalty = (cov - th.eye(logits.size(-1)).to(device)).sum() / 2
        loss = IC_loss(c, target)
        return loss + self.phi * penalty

# CCC loss function, a combination of MSE loss and IC loss
class CCC_loss(nn.Module):
    def __init__(self):
        super(CCC_loss, self).__init__()

    def forward(self, logits, target):
        logits_mean, target_mean = logits.mean(), target.mean()
        logits_var, target_var = ((logits - logits_mean) ** 2).sum(), ((target - target_mean) ** 2).sum()
        denomi = th.sum((logits - logits_mean) * (target - target_mean))
        ccc = 2 * denomi / (logits_var + target_var + (logits_mean - target_mean) ** 2)
        return -ccc

# CCC loss function, a combination of MSE loss and IC loss
class CCC_loss_penalty(nn.Module):
    def __init__(self, phi=1e-2):
        super(CCC_loss_penalty, self).__init__()
        self.phi = phi

    def forward(self, logits, target):
        logits_mean, target_mean = logits.mean(), target.mean()
        logits_var, target_var = ((logits - logits_mean) ** 2).sum(), ((target - target_mean) ** 2).sum()
        denomi = th.sum((logits - logits_mean) * (target - target_mean))
        ccc = 2 * denomi / (logits_var + target_var + (logits_mean - target_mean) ** 2)
        cov = th.corrcoef(logits.permute(1,0))
        penalty = (cov - th.eye(logits.size(-1)).to(device)).sum() / 2
        return -ccc + penalty * self.phi

# weighted ccc loss function
class WCCC_loss(nn.Module):
    def __init__(self, phi=1e-2):
        super(WCCC_loss, self).__init__()
        self.phi = phi

    def forward(self, logits, target):
        weight = th.argsort(th.argsort(target))
        weight = weight / th.max(weight)
        weight = th.exp((1 - weight) * th.log(th.tensor(0.5)) / TAU)
        weight = weight / th.sum(weight)
        logits_mean, target_mean = (weight * logits).sum(), (weight * target).sum()
        logits_var, target_var = (weight * (logits - logits_mean) ** 2).sum(), (weight * (target - target_mean) ** 2).sum()
        denomi = th.sum(weight * logits * target) - th.sum(weight * logits) * th.sum(weight * target)
        ccc = 2 * denomi / (logits_var + target_var + (logits_mean - target_mean) ** 2)
        cov = th.corrcoef(logits.permute(1,0))
        penalty = (cov - th.eye(logits.size(-1)).to(device)).sum() / 2
        return -ccc + penalty * self.phi
    
# weighted ccc loss function
class WCCC_loss_penalty(nn.Module):
    def __init__(self):
        super(WCCC_loss_penalty, self).__init__()

    def forward(self, logits, target):
        weight = th.argsort(th.argsort(target))
        weight = weight / th.max(weight)
        weight = th.exp((1 - weight) * th.log(th.tensor(0.5)) / TAU)
        weight = weight / th.sum(weight)
        logits_mean, target_mean = (weight * logits).sum(), (weight * target).sum()
        logits_var, target_var = (weight * (logits - logits_mean) ** 2).sum(), (weight * (target - target_mean) ** 2).sum()
        denomi = th.sum(weight * logits * target) - th.sum(weight * logits) * th.sum(weight * target)
        ccc = 2 * denomi / (logits_var + target_var + (logits_mean - target_mean) ** 2)
        return -ccc

class ListMLE(nn.Module):
    """
    ListMLE loss introduced in "Listwise Approach to Learning to Rank - Theory and Algorithm".
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param eps: epsilon value, used for numerical stability
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: loss value, a th.Tensor
    """
    def __init__(self):
        super(ListMLE, self).__init__()

    def forward(self, y_pred, y_true):
        # shuffle for randomised tie resolution
        random_indices = th.randperm(y_pred.shape[-1])
        y_pred_shuffled = y_pred[:, random_indices]
        y_true_shuffled = y_true[:, random_indices]

        y_true_sorted, indices = y_true_shuffled.sort(descending=True, dim=-1)

        mask = y_true_sorted == PADDED_Y_VALUE

        preds_sorted_by_true = th.gather(y_pred_shuffled, dim=1, index=indices)
        preds_sorted_by_true[mask] = float("-inf")

        max_pred_values, _ = preds_sorted_by_true.max(dim=1, keepdim=True)

        preds_sorted_by_true_minus_max = preds_sorted_by_true - max_pred_values

        cumsums = th.cumsum(preds_sorted_by_true_minus_max.exp().flip(dims=[1]), dim=1).flip(dims=[1])

        observation_loss = th.log(cumsums + DEFAULT_EPS) - preds_sorted_by_true_minus_max

        observation_loss[mask] = 0.0

        return th.mean(th.sum(observation_loss, dim=1))


class ListMLE_penalty(nn.Module):
    """
    ListMLE loss introduced in "Listwise Approach to Learning to Rank - Theory and Algorithm".
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param eps: epsilon value, used for numerical stability
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: loss value, a th.Tensor
    """
    def __init__(self, phi=1e-2):
        super(ListMLE_penalty, self).__init__()
        self.phi = phi

    def forward(self, x, y_true):
        y_pred = x.mean(dim=2)
        # shuffle for randomised tie resolution
        random_indices = th.randperm(y_pred.shape[-1])
        y_pred_shuffled = y_pred[:, random_indices]
        y_true_shuffled = y_true[:, random_indices]

        y_true_sorted, indices = y_true_shuffled.sort(descending=True, dim=-1)

        mask = y_true_sorted == PADDED_Y_VALUE

        preds_sorted_by_true = th.gather(y_pred_shuffled, dim=1, index=indices)
        preds_sorted_by_true[mask] = float("-inf")

        max_pred_values, _ = preds_sorted_by_true.max(dim=1, keepdim=True)

        preds_sorted_by_true_minus_max = preds_sorted_by_true - max_pred_values

        cumsums = th.cumsum(preds_sorted_by_true_minus_max.exp().flip(dims=[1]), dim=1).flip(dims=[1])

        observation_loss = th.log(cumsums + DEFAULT_EPS) - preds_sorted_by_true_minus_max

        observation_loss[mask] = 0.0

        l = th.mean(th.sum(observation_loss, dim=1))

        x = x.reshape((-1, x.size(2)))
        cov = th.corrcoef(x.permute(1,0))
        penalty = (cov - th.eye(x.size(-1)).to(device)).sum() / 2

        return l + penalty * self.phi

    
class ListNet(nn.Module):
    """
    ListNet loss introduced in "Learning to Rank: From Pairwise Approach to Listwise Approach".
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param eps: epsilon value, used for numerical stability
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: loss value, a th.Tensor
    """
    def __init__(self):
        super(ListNet, self).__init__()

    def forward(self, y_pred, y_true):
        y_pred = y_pred.clone()
        y_true = y_true.clone()

        mask = y_true == PADDED_Y_VALUE
        y_pred[mask] = float('-inf')
        y_true[mask] = float('-inf')

        preds_smax = F.softmax(y_pred, dim=1)
        true_smax = F.softmax(y_true, dim=1)

        preds_smax = preds_smax + DEFAULT_EPS
        preds_log = th.log(preds_smax)

        return th.mean(-th.sum(true_smax * preds_log, dim=1))
    

class ListNet_penalty(nn.Module):
    """
    ListNet loss introduced in "Learning to Rank: From Pairwise Approach to Listwise Approach".
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param eps: epsilon value, used for numerical stability
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: loss value, a th.Tensor
    """
    def __init__(self, phi=1e-2):
        super(ListNet_penalty, self).__init__()
        self.phi = phi

    def forward(self, x, y_true):
        y_pred = x.mean(dim=2)
        y_pred = y_pred.clone()
        y_true = y_true.clone()

        mask = y_true == PADDED_Y_VALUE
        y_pred[mask] = float('-inf')
        y_true[mask] = float('-inf')

        preds_smax = F.softmax(y_pred, dim=1)
        true_smax = F.softmax(y_true, dim=1)

        preds_smax = preds_smax + DEFAULT_EPS
        preds_log = th.log(preds_smax)

        l = th.mean(-th.sum(true_smax * preds_log, dim=1))

        x = x.reshape((-1, x.size(2)))
        cov = th.corrcoef(x.permute(1,0))
        penalty = (cov - th.eye(x.size(-1)).to(device)).sum() / 2

        return l + penalty * self.phi


class Closs(nn.Module):
    def __init__(self):
        super(Closs, self).__init__()
    def forward(self, f, num_stocks):
        l = th.sum(f[:,num_stocks // 2:], dim = 1) - th.sum(f[:, :num_stocks // 2], dim = 1)
        for i in range(num_stocks // 2):
            l += th.logsumexp(f[:,i:num_stocks-i], dim = 1)
            l += th.logsumexp(th.neg(f[:,i:num_stocks-i]), dim = 1)
        l = th.mean(l)
        return l
    
class Closs_penalty(nn.Module):
    def __init__(self, phi=1):
        super(Closs_penalty, self).__init__()
        self.phi = phi

    def forward(self, x, num_stocks):
        f = x.mean(dim = 2)
        l = th.sum(f[:,num_stocks // 2:], dim = 1) - th.sum(f[:, :num_stocks // 2], dim = 1)
        for i in range(num_stocks // 2):
            l += th.logsumexp(f[:,i:num_stocks-i], dim = 1)
            l += th.logsumexp(th.neg(f[:,i:num_stocks-i]), dim = 1)
        l = th.mean(l)
        x = x.reshape((-1, x.size(2)))
        cov = th.corrcoef(x.permute(1,0))
        penalty = (cov - th.eye(x.size(-1)).to(device)).sum() / 2
        return l + penalty * self.phi

class Closs_explained(nn.Module):
    def __init__(self):
        super(Closs_explained, self).__init__()
    def forward(self, f, num_stocks):
        l = th.sum(f[:,num_stocks // 2:], dim = 1) - th.sum(f[:, :num_stocks // 2], dim = 1)
        for i in range(num_stocks // 2):
            subtract = th.tensor(num_stocks - 2*i,requires_grad = False)
            l += th.log(th.sum(th.exp(f[:,i:num_stocks-i]), dim = 1)*th.sum(th.exp(th.neg(f[:,i:num_stocks-i])), dim = 1)-subtract)
        l = th.mean(l)
        return l

class Closs_explained_penalty(nn.Module):
    def __init__(self, phi=1):
        super(Closs_explained_penalty, self).__init__()
        self.phi = phi

    def forward(self, x, num_stocks):
        f = x.mean(dim = 2)
        l = th.sum(f[:,num_stocks // 2:], dim = 1) - th.sum(f[:, :num_stocks // 2], dim = 1)
        for i in range(num_stocks // 2):
            subtract = th.tensor(num_stocks - 2*i,requires_grad = False)
            l += th.log(th.sum(th.exp(f[:,i:num_stocks-i]), dim = 1)*th.sum(th.exp(th.neg(f[:,i:num_stocks-i])), dim = 1)-subtract)
        l = th.mean(l)
        x = x.reshape((-1, x.size(2)))
        cov = th.corrcoef(x.permute(1,0))
        penalty = (cov - th.eye(x.size(-1)).to(device)).sum() / 2
        return l + penalty * self.phi

class Closs_sigmoid(nn.Module):
    def __init__(self):
        super(Closs_sigmoid, self).__init__()
    def forward(self, f, num_stocks):
        l = th.tensor(1, requires_grad=False)+th.exp(f[:,num_stocks//2:] - f[:,:num_stocks//2])
        return th.mean(th.log(l))

class Closs_sigmoid_penalty(nn.Module):
    def __init__(self, phi=1):
        super(Closs_sigmoid_penalty, self).__init__()
        self.phi = phi

    def forward(self, x, num_stocks):
        f = x.mean(dim=2)
        l = th.tensor(1, requires_grad=False)+th.exp(f[:,num_stocks//2:] - f[:,:num_stocks//2])
        l = th.mean(th.log(l))
        x = x.reshape((-1, x.size(2)))
        cov = th.corrcoef(x.permute(1,0))
        penalty = (cov - th.eye(x.size(-1)).to(device)).sum() / 2
        return l + penalty * self.phi

class Lloss(nn.Module):
    def __init__(self):
        super(Lloss, self).__init__()
    def forward(self, f, num_stocks):
        l = th.neg(th.sum(f, dim = 1))
        for i in range(num_stocks):
            l += th.logsumexp(f[:,i:], dim = 1)
        l = th.mean(l)
        return l
    
class Lloss_penalty(nn.Module):
    def __init__(self, phi=1):
        super(Lloss_penalty, self).__init__()
        self.phi = phi

    def forward(self, x, num_stocks):
        f = x.mean(dim=2)
        l = th.neg(th.sum(f, dim = 1))
        for i in range(num_stocks):
            l += th.logsumexp(f[:,i:], dim = 1)
        l = th.mean(l)
        x = x.reshape((-1, x.size(2)))
        cov = th.corrcoef(x.permute(1,0))
        penalty = (cov - th.eye(x.size(-1)).to(device)).sum() / 2
        return l + penalty * self.phi

class TRR(nn.Module):
    """
    Temporal Relational Ranking
    """
    def __init__(self, alpha: float=10.0):
        super(TRR, self).__init__()
        self.alpha = alpha
        
    def forward(self, pred_y, true_y):
        pred_y = pred_y.unsqueeze(1).to(th.float32)
        true_y = true_y.unsqueeze(1).to(th.float32)
        reg_loss = nn.MSELoss()(pred_y, true_y)
        all_one = th.ones((pred_y.size(0), 1)).to(th.float32)
        pre_pw_dif = pred_y.matmul(all_one.permute(1,0)) - all_one.matmul(pred_y.permute(1,0))
        gt_pw_dif = true_y.matmul(all_one.permute(1,0)) - all_one.matmul(true_y.permute(1,0))
        rank_loss = (F.relu(pre_pw_dif * gt_pw_dif)).mean()
        loss = reg_loss + self.alpha * rank_loss
        return loss

# get the factor unit created by stock data
def get_stock_unit(save_dir: str, member_dir: str, start_date: int, end_date: int, train_window: int, bench: str):

    # create folder to store the unit data
    if not os.path.exists(f'{save_dir}/Unit/Stock/{bench}'):
        os.makedirs(f'{save_dir}/Unit/Stock/{bench}')

    # determine all required stock data files
    trDate = os.listdir(f'{save_dir}/Data/Stock/allAshare')
    trDate = np.array(trDate)
    trDate = trDate[(trDate >= f'{start_date}.csv') & (trDate <= f'{end_date}.csv')]

    # get the member data if necessary
    if bench != 'allAshare':
        member_df = read_member(member_dir, start_date, end_date)

    with trange(len(trDate)-train_window) as trDay_bar:    
        for i in trDay_bar:
            trDay_i = trDate[i+train_window]
            trDay_i = int(trDay_i[:8])
            # skip if data already exists
            if os.path.exists(f'{save_dir}/Unit/Stock/{bench}/{trDay_i}.pkl'):
                continue
            trDay_bar.set_description(f'Loading stock data on date {trDay_i}')
            # look back and load all the data
            start_i, end_i = trDate[i], trDate[i+train_window-1]
            dates_i = trDate[(trDate >= start_i) & (trDate <= end_i)]
            df_i = pd.DataFrame()
            for j in dates_i:
                stock_i = pd.read_csv(f'{save_dir}/Data/Stock/allAshare/{j}')
                df_i = df_i.append(stock_i)
            # select member if necessary
            if bench != 'allAshare':
                member_i = member_df.loc[member_df['date'] == int(dates_i[-1][:-4])]
                member_i = member_i.loc[:,['id']]
                df_i = pd.merge(df_i, member_i, on='id', how='inner')
            df_i.sort_values(by=['id', 'date'], inplace=True)
            df_i.set_index(keys=['id', 'date'], inplace=True)

            write_pkl5_data(f'{save_dir}/Unit/Stock/{bench}/{trDay_i}.pkl', df_i)

# get the factor unit created by factor data
def get_factor_unit(save_dir: str, factor_dir: str, member_dir: str, trDays_dir: str, start_date: int, end_date: int,
                     train_window: int, bench: str, name: str):

    # create folder to store the unit data
    if not os.path.exists(f'{save_dir}/Unit/Factor/{name}/{bench}'):
        os.makedirs(f'{save_dir}/Unit/Factor/{name}/{bench}')

    # retrieve the start and end trading date of the factor unit
    factorLst = os.listdir(factor_dir)
    startLst = [int(np.sort(os.listdir(f'{factor_dir}/{f}/allAshare'))[0][:8]) for f in factorLst]
    endLst = [int(np.sort(os.listdir(f'{factor_dir}/{f}/allAshare'))[-1][:8]) for f in factorLst]
    start, end = max(startLst+[start_date]), min(endLst+[end_date])

    # get the trading date sequence
    trDate = read_pkl5_data(trDays_dir)
    trDate = trDate.index.to_numpy()
    trDate = trDate[(trDate >= start) & (trDate <= end)]

    if bench != 'allAshare':
        member_df = read_member(member_dir, start_date, end_date)

    # delete factor if it has some missing data
    newfactorLst = []
    for factor in factorLst:
        dateLst = os.listdir(f'{factor_dir}/{factor}/allAshare')
        dateLst = np.array(dateLst)
        dateLst = dateLst[(dateLst >= f'{start}.csv') & (dateLst <= f'{end}.csv')]
        if len(dateLst) != len(trDate):
            print(len(dateLst))
            print(len(trDate))
            print(f'Factor {factor} has some missing data, we shall remove it')
        else:
            newfactorLst.append(factor)

    factorLst = newfactorLst
    print(f'Now we have {len(factorLst)} factors ...')

    # store factor unit data
    with trange(len(trDate)-train_window) as trDay_bar:
        for i in trDay_bar:
            trDay_i = trDate[i+train_window]
            trDay_bar.set_description(f'Loading factor data on date {trDay_i}')
            # skip if data already exists
            if os.path.exists(f'{save_dir}/Unit/Factor/{name}/{bench}/{trDay_i}.pkl'):
                continue
            # for each factor, look back and load all required data
            start_i, end_i = trDate[i], trDate[i+train_window-1]
            dates_i = trDate[(trDate >= start_i) & (trDate <= end_i)]
            df_i_Lst = []
            for factor in factorLst:
                df_i = pd.DataFrame()
                for j in dates_i:
                    factor_i = pd.read_csv(f'{factor_dir}/{factor}/allAshare/{j}.csv')
                    df_i = df_i.append(factor_i)
                if bench != 'allAshare':
                    member_i = member_df.loc[member_df['date'] == dates_i[-1]]
                    member_i = member_i.loc[:,['id']]
                    df_i = pd.merge(df_i, member_i, on='id', how='inner')
                df_i_Lst.append(df_i)

            df_i = reduce(lambda df1, df2: pd.merge(df1, df2, on=['id', 'date'], how='inner'), df_i_Lst)
            df_i.sort_values(by=['id', 'date'], inplace=True)
            df_i.set_index(keys=['id', 'date'], inplace=True)

            write_pkl5_data(f'{save_dir}/Unit/Factor/{name}/{bench}/{trDay_i}.pkl', df_i)

# load the features, label, and information of different datasets
def load_dataset(save_dir: str, dates: np.array, return_df: pd.DataFrame, bench: str, stock: str, factor1: str, factor2: str):

    stock_, factor1_, factor2_, label_, id_, tdate_ = [], [], [], [], [], []

    # load the training dataset
    with trange(len(dates)) as date_bar:
        for i in date_bar:
            date_i = dates[i]
            date_bar.set_description(f'Loading data on date {date_i}')

            # get data from different datasets
            data_i = []

            if not stock is None:
                if not os.path.exists(f'{save_dir}/Unit/{stock}/{bench}/{date_i}.pkl'):
                    continue
                stock_i = read_pkl5_data(f'{save_dir}/Unit/{stock}/{bench}/{date_i}.pkl')
                data_i.append(stock_i)

            if not factor1 is None:
                if not os.path.exists(f'{save_dir}/Unit/Factor/{factor1}/{bench}/{date_i}.pkl'):
                    continue
                factor1_i = read_pkl5_data(f'{save_dir}/Unit/Factor/{factor1}/{bench}/{date_i}.pkl')
                data_i.append(factor1_i)

            if not factor2 is None:
                if not os.path.exists(f'{save_dir}/Unit/Factor/{factor2}/{bench}/{date_i}.pkl'):
                    continue
                factor2_i = read_pkl5_data(f'{save_dir}/Unit/Factor/{factor2}/{bench}/{date_i}.pkl')
                data_i.append(factor2_i)

            return_i = return_df.loc[return_df['date'] == date_i]
            return_i = return_i.loc[:,['id', 'return']]

            if len(data_i) > 1:
                df_i = reduce(lambda df1, df2: pd.merge(df1, df2, on=['id', 'date'], how='inner'), data_i)
            elif len(data_i) == 1:
                df_i = data_i[0]

            df_i.reset_index(drop=False, inplace=True)
            df_i = pd.merge(df_i, return_i, on='id', how='inner')
            df_i.set_index(['id', 'date'], inplace=True)

            stocklst_i, factor1lst_i, factor2lst_i = [], [], []

            # stack all the data and create th tensor
            if not stock is None:
                for col in [c for c in stock_i.columns if not c in ['id', 'date']]:
                    temp_i = df_i.loc[:,col]
                    temp_i = temp_i.unstack()
                    features = copy.deepcopy(temp_i.values)
                    # fill nan with column average
                    col_mean = np.nanmean(features, axis=0)
                    inds = np.where(np.isnan(features))
                    features[inds] = np.take(col_mean, inds[1])
                    # clip extreme value with 5 MAD
                    MAD = np.median(np.abs(features-np.median(features, axis=0)[None,:]), axis=0)
                    med = np.median(features, axis=0)
                    for i in range(len(med)):
                        features[:,i] = np.clip(features[:,i], med[i]-5*MAD[i], med[i]+5*MAD[i])
                    # normalization
                    features = sp.stats.zscore(features, axis=0)
                    stocklst_i.append(features)
                stock_i = np.stack(stocklst_i, axis=2)
                stock_.append(stock_i)

            if not factor1 is None:
                for col in [c for c in factor1_i.columns if not c in ['id', 'date']]:
                    temp_i = df_i.loc[:,col]
                    temp_i = temp_i.unstack()
                    features = copy.deepcopy(temp_i.values)
                    # fill nan with column average
                    col_mean = np.nanmean(features, axis=0)
                    inds = np.where(np.isnan(features))
                    features[inds] = np.take(col_mean, inds[1])
                    # clip extreme value with 5 MAD
                    MAD = np.median(np.abs(features-np.median(features, axis=0)[None,:]), axis=0)
                    med = np.median(features, axis=0)
                    for i in range(len(med)):
                        features[:,i] = np.clip(features[:,i], med[i]-5*MAD[i], med[i]+5*MAD[i])
                    # normalization
                    features = sp.stats.zscore(features, axis=0)
                    factor1lst_i.append(features)
                factor1_i = np.stack(factor1lst_i, axis=2)
                factor1_.append(factor1_i)

            if not factor2 is None:
                for col in [c for c in factor2_i.columns if not c in ['id', 'date']]:
                    temp_i = df_i.loc[:,col]
                    temp_i = temp_i.unstack()
                    features = copy.deepcopy(temp_i.values)
                    # fill nan with column average
                    col_mean = np.nanmean(features, axis=0)
                    inds = np.where(np.isnan(features))
                    features[inds] = np.take(col_mean, inds[1])
                    # clip extreme value with 5 MAD
                    MAD = np.median(np.abs(features-np.median(features, axis=0)[None,:]), axis=0)
                    med = np.median(features, axis=0)
                    for i in range(len(med)):
                        features[:,i] = np.clip(features[:,i], med[i]-5*MAD[i], med[i]+5*MAD[i])
                    # normalization
                    features = sp.stats.zscore(features, axis=0)
                    factor2lst_i.append(features)
                factor2_i = np.stack(factor2lst_i, axis=2)
                factor2_.append(factor2_i)

            df_i.reset_index(drop=False, inplace=True)

            # get the labels of the dataset
            df_i = df_i.loc[:,['id', 'return']].drop_duplicates(['id', 'return'])
            id_i = df_i['id']
            id_i = id_i.values
            tdate_i = np.array([date_i] * id_i.shape[0])
            return_i = df_i['return']
            return_i = return_i.values

            label_.append(return_i)
            id_.append(id_i)
            tdate_.append(tdate_i)

    # concatenate all the tensor data
    if not stock is None:
        stock_ = np.concatenate(stock_, axis=0)

    if not factor1 is None:
        factor1_ = np.concatenate(factor1_, axis=0)

    if not factor2 is None:
        factor2_ = np.concatenate(factor2_, axis=0)
        
    label_ = np.concatenate(label_, axis=0)
    id_ = np.concatenate(id_, axis=0)
    tdate_ = np.concatenate(tdate_, axis=0)

    return stock_, factor1_, factor2_, label_, id_, tdate_

# load the features, label, and information of different datasets
def load_dataset_rank(save_dir: str, dates: np.array, return_df: pd.DataFrame, member_df: pd.DataFrame, stock: str, 
                      factor1: str, factor2: str, window: int, n_stock: int, member: str):

    stock_, factor1_, factor2_ = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    if not member_df is None and member == 'topN':
        stock_id, factor1_id, factor2_id = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        with trange(len(dates)) as date_bar:
            for i in date_bar:
                date_i = dates[i]
                date_bar.set_description(f'Determine  {date_i}')

                if not stock is None:
                    if not os.path.exists(f'{save_dir}/Unit/{stock}/allAshare/{date_i}.pkl'):
                        continue
                    stock_i = read_pkl5_data(f'{save_dir}/Unit/{stock}/allAshare/{date_i}.pkl')
                    stock_i.reset_index(drop=False, inplace=True)
                    stock_id_i = pd.DataFrame({'id': pd.unique(stock_i['id'])})
                    stock_id = stock_id.append(stock_id_i)

                if not factor1 is None:
                    if not os.path.exists(f'{save_dir}/Unit/Factor/{factor1}/allAshare/{date_i}.pkl'):
                        continue
                    factor1_i = read_pkl5_data(f'{save_dir}/Unit/Factor/{factor1}/allAshare/{date_i}.pkl')
                    factor1_i.reset_index(drop=False, inplace=True)
                    factor1_id_i = pd.DataFrame({'id': pd.unique(factor1_i['id'])})
                    factor1_id = factor1_id.append(factor1_id_i)

                if not factor2 is None:
                    if not os.path.exists(f'{save_dir}/Unit/Factor/{factor2}/allAshare/{date_i}.pkl'):
                        continue
                    factor2_i = read_pkl5_data(f'{save_dir}/Unit/Factor/{factor2}/allAshare/{date_i}.pkl')
                    factor2_i.reset_index(drop=False, inplace=True)
                    factor2_id_i = pd.DataFrame({'id': pd.unique(factor2_i['id'])})
                    factor2_id = factor2_id.append(factor2_id_i)

        return_id = pd.DataFrame({'id': pd.unique(return_df['id'])})
        stock_id = pd.DataFrame({'id': pd.unique(stock_id['id'])})
        factor1_id = pd.DataFrame({'id': pd.unique(factor1_id['id'])})
        factor2_id = pd.DataFrame({'id': pd.unique(factor2_id['id'])})
        member_df = pd.merge(return_id, member_df, on='id', how='inner')
        member_df = pd.merge(stock_id, member_df, on='id', how='inner')
        member_df = pd.merge(factor1_id, member_df, on='id', how='inner')
        member_df = pd.merge(factor2_id, member_df, on='id', how='inner')
        member_df.sort_values(by='cnt', inplace=True)
        member_df = member_df.iloc[-n_stock:,:].reset_index(drop=True)

    # load the training dataset
    with trange(len(dates)) as date_bar:
        for i in date_bar:
            date_i = dates[i]
            date_bar.set_description(f'Loading data on date {date_i}')

            if not stock is None:
                if not os.path.exists(f'{save_dir}/Unit/{stock}/allAshare/{date_i}.pkl'):
                    continue
                stock_i = read_pkl5_data(f'{save_dir}/Unit/{stock}/allAshare/{date_i}.pkl')
                stock_i = stock_i.unstack()
                stock_i = stock_i.stack(dropna=False)
                cols = stock_i.columns
                stock_i.reset_index(drop=False, inplace=True)
                if not member_df is None and member == 'topN':
                    stock_i = pd.merge(member_df, stock_i, on='id', how='left')
                    del stock_i['cnt']
                elif not member_df is None and member == 'bench':
                    member_i = member_df.loc[member_df['date'] == date_i]
                    stock_i = pd.merge(member_i['id'], stock_i, on='id', how='left')
                idLst = pd.unique(stock_i['id'])
                stock_i.set_index(['id', 'date'], inplace=True)
                features = stock_i.values.reshape(len(idLst), -1)
                features = np.float32(features)
                stock_i = pd.DataFrame({'id': idLst})
                stock_i['date'] = date_i
                new_cols = [f'{j}_{i}' for i in range(60) for j in cols]
                stock_i[new_cols] = features
                stock_ = stock_.append(stock_i)

            if not factor1 is None:
                if not os.path.exists(f'{save_dir}/Unit/Factor/{factor1}/allAshare/{date_i}.pkl'):
                    continue
                factor1_i = read_pkl5_data(f'{save_dir}/Unit/Factor/{factor1}/allAshare/{date_i}.pkl')
                factor1_i = factor1_i.unstack()
                factor1_i = factor1_i.stack(dropna=False)
                cols = factor1_i.columns
                factor1_i.reset_index(drop=False, inplace=True)
                if not member_df is None and member == 'topN':
                    factor1_i = pd.merge(member_df, factor1_i, on='id', how='left')
                    del factor1_i['cnt']
                elif not member_df is None and member == 'bench':
                    member_i = member_df.loc[member_df['date'] == date_i]
                    factor1_i = pd.merge(member_i['id'], factor1_i, on='id', how='left')
                idLst = pd.unique(factor1_i['id'])
                factor1_i.set_index(['id', 'date'], inplace=True)
                features = factor1_i.values.reshape(len(idLst), -1)
                features = np.float32(features)
                factor1_i = pd.DataFrame({'id': idLst})
                factor1_i['date'] = date_i
                new_cols = [f'{j}_{i}' for i in range(60) for j in cols]
                factor1_i[new_cols] = features
                factor1_ = factor1_.append(factor1_i)

            if not factor2 is None:
                if not os.path.exists(f'{save_dir}/Unit/Factor/{factor2}/allAshare/{date_i}.pkl'):
                    continue
                factor2_i = read_pkl5_data(f'{save_dir}/Unit/Factor/{factor2}/allAshare/{date_i}.pkl')
                factor2_i = factor2_i.unstack()
                factor2_i = factor2_i.stack(dropna=False)
                cols = factor2_i.columns
                factor2_i.reset_index(drop=False, inplace=True)
                if not member_df is None and member == 'topN':
                    factor2_i = pd.merge(member_df, factor2_i, on='id', how='left')
                    del factor2_i['cnt']
                elif not member_df is None and member == 'bench':
                    member_i = member_df.loc[member_df['date'] == date_i]
                    factor2_i = pd.merge(member_i['id'], factor2_i, on='id', how='left')
                idLst = pd.unique(factor2_i['id'])
                factor2_i.set_index(['id', 'date'], inplace=True)
                features = factor2_i.values.reshape(len(idLst), -1)
                features = np.float32(features)
                factor2_i = pd.DataFrame({'id': idLst})
                factor2_i['date'] = date_i
                new_cols = [f'{j}_{i}' for i in range(60) for j in cols]
                factor2_i[new_cols] = features
                factor2_ = factor2_.append(factor2_i)

    stock_.set_index(['id', 'date'], inplace=True)
    factor1_.set_index(['id', 'date'], inplace=True)
    factor2_.set_index(['id', 'date'], inplace=True)

    df = pd.merge(stock_, factor1_, on=['id', 'date'], how='inner')
    df = pd.merge(df, factor2_, on=['id', 'date'], how='inner')
    df.sort_index(inplace=True)

    stocklst, factor1lst, factor2lst = [], [], []

    # stack all the data and create th tensor
    if not stock is None:
        for col in [c for c in stock_.columns if not c in ['id', 'date']]:
            temp_i = df.loc[:,col]
            temp_i = temp_i.unstack()
            features = copy.deepcopy(temp_i.values)
            # fill nan with column average
            col_mean = np.nanmean(features, axis=0)
            inds = np.where(np.isnan(features))
            features[inds] = np.take(col_mean, inds[1])
            # clip extreme value with 5 MAD
            MAD = np.median(np.abs(features-np.median(features, axis=0)[None,:]), axis=0)
            med = np.median(features, axis=0)
            for i in range(len(med)):
                features[:,i] = np.clip(features[:,i], med[i]-5*MAD[i], med[i]+5*MAD[i])
            # normalization
            features = sp.stats.zscore(features, axis=0)
            stocklst.append(features)
        stock_ = np.stack(stocklst, axis=2)

    if not factor1 is None:
        for col in [c for c in factor1_.columns if not c in ['id', 'date']]:
            temp_i = df.loc[:,col]
            temp_i = temp_i.unstack()
            features = copy.deepcopy(temp_i.values)
            # fill nan with column average
            col_mean = np.nanmean(features, axis=0)
            inds = np.where(np.isnan(features))
            features[inds] = np.take(col_mean, inds[1])
            # clip extreme value with 5 MAD
            MAD = np.median(np.abs(features-np.median(features, axis=0)[None,:]), axis=0)
            med = np.median(features, axis=0)
            for i in range(len(med)):
                features[:,i] = np.clip(features[:,i], med[i]-5*MAD[i], med[i]+5*MAD[i])
            # normalization
            features = sp.stats.zscore(features, axis=0)
            factor1lst.append(features)
        factor1_ = np.stack(factor1lst, axis=2)

    if not factor2 is None:
        for col in [c for c in factor2_.columns if not c in ['id', 'date']]:
            temp_i = df.loc[:,col]
            temp_i = temp_i.unstack()
            features = copy.deepcopy(temp_i.values)
            # fill nan with column average
            col_mean = np.nanmean(features, axis=0)
            inds = np.where(np.isnan(features))
            features[inds] = np.take(col_mean, inds[1])
            # clip extreme value with 5 MAD
            MAD = np.median(np.abs(features-np.median(features, axis=0)[None,:]), axis=0)
            med = np.median(features, axis=0)
            for i in range(len(med)):
                features[:,i] = np.clip(features[:,i], med[i]-5*MAD[i], med[i]+5*MAD[i])
            # normalization
            features = sp.stats.zscore(features, axis=0)
            factor2lst.append(features)
        factor2_ = np.stack(factor2lst, axis=2)

    df.reset_index(drop=False, inplace=True)
    df = pd.merge(df, return_df, on=['id', 'date'])

    # get the labels of the dataset
    label_ = df['return'].values
    id_ = df['id'].values
    tdate_ = df['date'].values

    return stock_, factor1_, factor2_, label_, id_, tdate_

# train and test data in the order of year
def train_and_test(save_dir: str, trDays_dir: str, modelLst: list, loss: str, start_year: int, end_year: int, train_window: int, 
                   eval_window: int, test_window: int, n_factors: int, back_window: int, bench: str, T: int, stock: str, factor1: str, factor2: str,
                   gap: bool, save_npy: bool):
    
    # create the folder to save model and rolling data
    if not os.path.exists(f'{save_dir}/Model/{bench}'):
        os.makedirs(f'{save_dir}/Model/{bench}')

    if not os.path.exists(f'{save_dir}/Rolling/{bench}'):
        os.makedirs(f'{save_dir}/Rolling/{bench}')

    # get the trading date data within this time interval
    timeLst = read_pkl5_data(trDays_dir)
    timeLst = timeLst.index.to_numpy()
    timeLst = timeLst[(int(f'{start_year-1}0101') <= timeLst) & (timeLst <= int(f'{end_year}1231'))]

    # # load the stock return data
    # return_df = pd.DataFrame()

    # with trange(len(timeLst)) as date_bar:
    #     for i in date_bar:
    #         date_i = timeLst[i]
    #         date_bar.set_description(f'Loading data on date {date_i}')

    #         if gap:
    #             try:
    #                 return_i = pd.read_csv(f'{save_dir}/Data/Return/allAshare/T+1_T+{T+1}/{date_i}.csv')
    #             except:
    #                 print(f'No such file: {save_dir}/Data/Return/allAshare/T+1_T+{T+1}/{date_i}.csv')
    #                 continue
    #         else:
    #             try:
    #                 return_i = pd.read_csv(f'{save_dir}/Data/Return/allAshare/T_T+{T}/{date_i}.csv')
    #             except:
    #                 print(f'No such file: {save_dir}/Data/Return/allAshare/T_T+{T}/{date_i}.csv')

    #         return_df = return_df.append(return_i)

    # iterately train and test the model
    for year_i in range(start_year, end_year-train_window-eval_window-test_window+2):

        print(f'''processing data with training period {year_i}-{year_i+train_window-1}
        evaluation period {year_i+train_window}-{year_i+train_window+eval_window-1}
        and testing period {year_i+train_window+eval_window}-{year_i+train_window+eval_window+test_window-1}''')

        # get the dates for corresponding data sets
        train_dates = timeLst[(int(f'{year_i}0101') <= timeLst) &
             (timeLst <= int(f'{year_i+train_window-1}1231'))]
        eval_dates = timeLst[(int(f'{year_i+train_window}0101') <= timeLst) &
             (timeLst <= int(f'{year_i+train_window+eval_window-1}1231'))]
        test_dates = timeLst[(int(f'{year_i+train_window+eval_window}0101') <= timeLst) &
             (timeLst <= int(f'{year_i+train_window+eval_window+test_window-1}1231'))]

        # load training and testing datasets
        if save_npy:
            if os.path.exists(f'{save_dir}/Rolling/{bench}/{train_dates[0]}_{train_dates[-1]}_stock.npy'):
                # train_stock, train_factor1, train_factor2, train_Y, train_id, train_tdate = np.load(f'{save_dir}/Rolling/{bench}/{train_dates[0]}_{train_dates[-1]}_data.npy', allow_pickle=True)
                train_stock = np.load(f'{save_dir}/Rolling/{bench}/{train_dates[0]}_{train_dates[-1]}_stock.npy', allow_pickle=True)
                train_factor1 = np.load(f'{save_dir}/Rolling/{bench}/{train_dates[0]}_{train_dates[-1]}_factor1.npy', allow_pickle=True)
                train_factor2 = np.load(f'{save_dir}/Rolling/{bench}/{train_dates[0]}_{train_dates[-1]}_factor2.npy', allow_pickle=True)
                train_Y = np.load(f'{save_dir}/Rolling/{bench}/{train_dates[0]}_{train_dates[-1]}_label.npy', allow_pickle=True)
                train_id = np.load(f'{save_dir}/Rolling/{bench}/{train_dates[0]}_{train_dates[-1]}_id.npy', allow_pickle=True)
                train_tdate = np.load(f'{save_dir}/Rolling/{bench}/{train_dates[0]}_{train_dates[-1]}_date.npy', allow_pickle=True)
            else:
                train_stock, train_factor1, train_factor2, train_Y, train_id, train_tdate = load_dataset(save_dir, train_dates, return_df, bench, stock, factor1, factor2)
                # train_data = np.empty(6, dtype=object)
                # train_data[:] = [train_stock, train_factor1, train_factor2, train_Y, train_id, train_tdate]
                # np.save(f'{save_dir}/Rolling/{bench}/{train_dates[0]}_{train_dates[-1]}_data.npy', train_data, allow_pickle=True)
                np.save(f'{save_dir}/Rolling/{bench}/{train_dates[0]}_{train_dates[-1]}_stock.npy', train_stock, allow_pickle=True)
                np.save(f'{save_dir}/Rolling/{bench}/{train_dates[0]}_{train_dates[-1]}_factor1.npy', train_factor1, allow_pickle=True)
                np.save(f'{save_dir}/Rolling/{bench}/{train_dates[0]}_{train_dates[-1]}_factor2.npy', train_factor2, allow_pickle=True)
                np.save(f'{save_dir}/Rolling/{bench}/{train_dates[0]}_{train_dates[-1]}_label.npy', train_Y, allow_pickle=True)
                np.save(f'{save_dir}/Rolling/{bench}/{train_dates[0]}_{train_dates[-1]}_id.npy', train_id, allow_pickle=True)
                np.save(f'{save_dir}/Rolling/{bench}/{train_dates[0]}_{train_dates[-1]}_date.npy', train_tdate, allow_pickle=True)
        else:
            train_stock, train_factor1, train_factor2, train_Y, train_id, train_tdate = load_dataset(save_dir, train_dates, return_df, bench, stock, factor1, factor2)

        train_Y = np.nan_to_num(train_Y, nan=0)

        if save_npy:
            if os.path.exists(f'{save_dir}/Rolling/{bench}/{test_dates[0]}_{test_dates[-1]}_stock.npy'):
                # test_stock, test_factor1, test_factor2, test_Y, test_id, test_tdate = np.load(f'{save_dir}/Rolling/{bench}/{test_dates[0]}_{test_dates[-1]}_data.npy', allow_pickle=True)
                test_stock = np.load(f'{save_dir}/Rolling/{bench}/{test_dates[0]}_{test_dates[-1]}_stock.npy', allow_pickle=True)
                test_factor1 = np.load(f'{save_dir}/Rolling/{bench}/{test_dates[0]}_{test_dates[-1]}_factor1.npy', allow_pickle=True)
                test_factor2 = np.load(f'{save_dir}/Rolling/{bench}/{test_dates[0]}_{test_dates[-1]}_factor2.npy', allow_pickle=True)
                test_Y = np.load(f'{save_dir}/Rolling/{bench}/{test_dates[0]}_{test_dates[-1]}_label.npy', allow_pickle=True)
                test_id = np.load(f'{save_dir}/Rolling/{bench}/{test_dates[0]}_{test_dates[-1]}_id.npy', allow_pickle=True)
                test_tdate = np.load(f'{save_dir}/Rolling/{bench}/{test_dates[0]}_{test_dates[-1]}_date.npy', allow_pickle=True)
            else:
                test_stock, test_factor1, test_factor2, test_Y, test_id, test_tdate = load_dataset(save_dir, test_dates, return_df, bench, stock, factor1, factor2)
                # test_data = np.empty(6, dtype=object)
                # test_data[:] = [test_stock, test_factor1, test_factor2, test_Y, test_id, test_tdate]
                # np.save(f'{save_dir}/Rolling/{bench}/{test_dates[0]}_{test_dates[-1]}_data.npy', test_data, allow_pickle=True)
                np.save(f'{save_dir}/Rolling/{bench}/{test_dates[0]}_{test_dates[-1]}_stock.npy', test_stock, allow_pickle=True)
                np.save(f'{save_dir}/Rolling/{bench}/{test_dates[0]}_{test_dates[-1]}_factor1.npy', test_factor1, allow_pickle=True)
                np.save(f'{save_dir}/Rolling/{bench}/{test_dates[0]}_{test_dates[-1]}_factor2.npy', test_factor2, allow_pickle=True)
                np.save(f'{save_dir}/Rolling/{bench}/{test_dates[0]}_{test_dates[-1]}_label.npy', test_Y, allow_pickle=True)
                np.save(f'{save_dir}/Rolling/{bench}/{test_dates[0]}_{test_dates[-1]}_id.npy', test_id, allow_pickle=True)
                np.save(f'{save_dir}/Rolling/{bench}/{test_dates[0]}_{test_dates[-1]}_date.npy', test_tdate, allow_pickle=True)
        else:
            test_stock, test_factor1, test_factor2, test_Y, test_id, test_tdate = load_dataset(save_dir, test_dates, return_df, bench, stock, factor1, factor2)

        test_Y = np.nan_to_num(test_Y, nan=0)

        # load the evaluation data set if it exists
        if len(eval_dates) != 0:
            if save_npy:
                if os.path.exists(f'{save_dir}/Rolling/{bench}/{eval_dates[0]}_{eval_dates[-1]}_stock.npy'):
                    # eval_stock, eval_factor1, eval_factor2, eval_Y, eval_id, eval_tdate = np.load(f'{save_dir}/Rolling/{bench}/{eval_dates[0]}_{eval_dates[-1]}_data.npy', allow_pickle=True)
                    eval_stock = np.load(f'{save_dir}/Rolling/{bench}/{eval_dates[0]}_{eval_dates[-1]}_stock.npy', allow_pickle=True)
                    eval_factor1 = np.load(f'{save_dir}/Rolling/{bench}/{eval_dates[0]}_{eval_dates[-1]}_factor1.npy', allow_pickle=True)
                    eval_factor2 = np.load(f'{save_dir}/Rolling/{bench}/{eval_dates[0]}_{eval_dates[-1]}_factor2.npy', allow_pickle=True)
                    eval_Y = np.load(f'{save_dir}/Rolling/{bench}/{eval_dates[0]}_{eval_dates[-1]}_label.npy', allow_pickle=True)
                    eval_id = np.load(f'{save_dir}/Rolling/{bench}/{eval_dates[0]}_{eval_dates[-1]}_id.npy', allow_pickle=True)
                    eval_tdate = np.load(f'{save_dir}/Rolling/{bench}/{eval_dates[0]}_{eval_dates[-1]}_date.npy', allow_pickle=True)
                else:
                    eval_stock, eval_factor1, eval_factor2, eval_Y, eval_id, eval_tdate = load_dataset(save_dir, eval_dates, return_df, bench, stock, factor1, factor2)
                    # eval_data = np.empty(6, dtype=object)
                    # eval_data[:] = [eval_stock, eval_factor1, eval_factor2, eval_Y, eval_id, eval_tdate]
                    # np.save(f'{save_dir}/Rolling/{bench}/{eval_dates[0]}_{eval_dates[-1]}_data.npy', eval_data, allow_pickle=True)
                    np.save(f'{save_dir}/Rolling/{bench}/{eval_dates[0]}_{eval_dates[-1]}_stock.npy', eval_stock, allow_pickle=True)
                    np.save(f'{save_dir}/Rolling/{bench}/{eval_dates[0]}_{eval_dates[-1]}_factor1.npy', eval_factor1, allow_pickle=True)
                    np.save(f'{save_dir}/Rolling/{bench}/{eval_dates[0]}_{eval_dates[-1]}_factor2.npy', eval_factor2, allow_pickle=True)
                    np.save(f'{save_dir}/Rolling/{bench}/{eval_dates[0]}_{eval_dates[-1]}_label.npy', eval_Y, allow_pickle=True)
                    np.save(f'{save_dir}/Rolling/{bench}/{eval_dates[0]}_{eval_dates[-1]}_id.npy', eval_id, allow_pickle=True)
                    np.save(f'{save_dir}/Rolling/{bench}/{eval_dates[0]}_{eval_dates[-1]}_date.npy', eval_tdate, allow_pickle=True)
            else:
                eval_stock, eval_factor1, eval_factor2, eval_Y, eval_id, eval_tdate = load_dataset(save_dir, eval_dates, return_df, bench, stock, factor1, factor2)

            eval_Y = np.nan_to_num(eval_Y, nan=0)

        # create data loaders for training and testing datasets
        if not stock is None:
            n_feature_stock = train_stock.shape[-1]
            train_stock, test_stock = np.nan_to_num(train_stock, nan=0), np.nan_to_num(test_stock, nan=0)
            train_stock, test_stock = th.from_numpy(train_stock), th.from_numpy(test_stock)
            train_stock, test_stock = train_stock.to(th.float32), test_stock.to(th.float32)
            train_stock_dataset = Data(train_stock, train_Y)
            test_stock_dataset = Data(test_stock, test_Y)
            train_stock_dataloader = DataLoader(train_stock_dataset, BATCH_SIZE, shuffle=False)
            test_stock_dataloader = DataLoader(test_stock_dataset, BATCH_SIZE, shuffle=False)

            # train the model one by one
            for model_name in modelLst:
                # train the model if and only if we haven't trained it before
                if os.path.exists(f'{save_dir}/Model/{bench}/{model_name}_{loss}_stock_{year_i}_{year_i+train_window-1}.m'):
                    print(f'Model {model_name} for stock already exists, loading model ...')
                    Best_model_stock = joblib.load(f'{save_dir}/Model/{bench}/{model_name}_{loss}_stock_{year_i}_{year_i+train_window-1}.m')
                    Best_model_stock.to(device=device)
                    Best_model_stock.eval()

                else:
                    if model_name == 'LSTM':
                        model = LSTMModel(input_size=n_feature_stock, output_size=n_factors)
                    elif model_name == 'GRU':
                        model = GRUModel(input_size=n_feature_stock, output_size=n_factors)
                    elif model_name == 'ALSTM':
                        model = ALSTMModel(input_size=n_feature_stock, output_size=n_factors)
                    elif model_name == 'TCN':
                        model = TCNModel(num_input=back_window, output_size=n_factors, num_feature=n_feature_stock)
                    elif model_name == 'Transformer':
                        model = Transformer(input_size=n_feature_stock, output_size=n_factors)
                    else:
                        raise ValueError(f'The parameter model should be LSTM/GRU/ALSTM/TCN/Transformer, get {model_name} instead')
                    
                    if loss == 'IC':
                        loss_func = IC_loss()
                    elif loss == 'IC_penalty':
                        loss_func = IC_loss_penalty()
                    elif loss == 'CCC':
                        loss_func = CCC_loss()
                    elif loss == 'WCCC':
                        loss_func = WCCC_loss()
                    elif loss == 'TRR':
                        loss_func = TRR()
                    else:
                        raise ValueError(f'The paramter loss should be IC/IC_penalty, get {loss} instead')
                    
                    print(f'Model {model_name} for stock does not exist, training model ...')
                    model.to(device=device)
                    model.train()

                    if len(eval_dates) != 0:
                        eval_stock = th.from_numpy(eval_stock)
                        eval_stock = eval_stock.to(th.float32)
                        eval_stock_dataset = Data(eval_stock, train_Y)
                        eval_stock_dataloader = DataLoader(eval_stock_dataset, BATCH_SIZE, shuffle=False)
                        Best_model_stock, _ = train(model, train_stock_dataloader, loss_func, eval_stock_dataloader, MAX_EPOCH)
                    else:
                        Best_model_stock, _ = train(model, train_stock_dataloader, loss_func, None, MAX_EPOCH)

                    joblib.dump(Best_model_stock, f'{save_dir}/Model/{bench}/{model_name}_{loss}_stock_{year_i}_{year_i+train_window-1}.m')

                # get the corresponding factor data
                predLst_stock = get_factor(Best_model_stock, test_stock_dataloader)

                if len(predLst_stock.shape) == 1:
                    stock_dict = {'id': test_id, 'date': test_tdate, 'return': test_Y}
                    stock_dict[f'{model_name}_stock'] = predLst_stock
                elif len(predLst_stock.shape) == 2:
                    stock_dict = {'id': test_id, 'date': test_tdate, 'return': test_Y}
                    for i in range(predLst_stock.shape[1]):
                        stock_dict[f'{model_name}_stock_{i+1}'] = predLst_stock[:,i]

                stock_df = pd.DataFrame(stock_dict)

                if not os.path.exists(f'{save_dir}/Factor/{loss}/{bench}/Stock/{model_name}'):
                    os.makedirs(f'{save_dir}/Factor/{loss}/{bench}/Stock/{model_name}')

                # store the factor data
                stock_tdate = np.sort(pd.unique(stock_df['date']))
                with trange(len(stock_tdate)) as date_bar:
                    for i in date_bar:
                        date_i = stock_tdate[i]
                        date_bar.set_description(f'Saving data on date {date_i}')

                        if not stock is None:
                            stock_df_i = stock_df.loc[stock_df['date'] == date_i]
                            stock_df_i.to_csv(f'{save_dir}/Factor/{loss}/{bench}/Stock/{model_name}/{date_i}.csv', index=False)

        if not factor1 is None:
            n_feature_factor1 = train_factor1.shape[-1]
            train_factor1, test_factor1 = np.nan_to_num(train_factor1, nan=0), np.nan_to_num(test_factor1, nan=0)
            train_factor1, test_factor1 = th.from_numpy(train_factor1), th.from_numpy(test_factor1)
            train_factor1, test_factor1 = train_factor1.to(th.float32), test_factor1.to(th.float32)
            train_factor1_dataset = Data(train_factor1, train_Y)
            test_factor1_dataset = Data(test_factor1, test_Y)
            train_factor1_dataloader = DataLoader(train_factor1_dataset, BATCH_SIZE, shuffle=False)
            test_factor1_dataloader = DataLoader(test_factor1_dataset, BATCH_SIZE, shuffle=False)

            # train the model one by one
            for model_name in modelLst:
                # train the model if and only if we haven't trained it before
                if os.path.exists(f'{save_dir}/Model/{bench}/{model_name}_{loss}_factor1_{year_i}_{year_i+train_window-1}.m'):
                    print(f'Model {model_name} for factor 1 already exists, loading model ...')
                    Best_model_factor1 = joblib.load(f'{save_dir}/Model/{bench}/{model_name}_{loss}_factor1_{year_i}_{year_i+train_window-1}.m')
                    Best_model_factor1.to(device=device)
                    Best_model_factor1.eval()

                else:
                    if model_name == 'LSTM':
                        model = LSTMModel(input_size=n_feature_factor1, output_size=n_factors)
                    elif model_name == 'GRU':
                        model = GRUModel(input_size=n_feature_factor1, output_size=n_factors)
                    elif model_name == 'ALSTM':
                        model = ALSTMModel(input_size=n_feature_factor1, output_size=n_factors)
                    elif model_name == 'TCN':
                        model = TCNModel(num_input=back_window, output_size=n_factors, num_feature=n_feature_factor1)
                    elif model_name == 'Transformer':
                        model = Transformer(input_size=n_feature_factor1, output_size=n_factors)
                    else:
                        raise ValueError(f'The parameter model should be LSTM/GRU/ALSTM/TCN/Transformer, get {model_name} instead')
                    
                    if loss == 'IC':
                        loss_func = IC_loss
                    elif loss == 'IC_penalty':
                        loss_func = IC_loss_penalty
                    elif loss == 'CCC':
                        loss_func = CCC_loss
                    elif loss == 'WCCC':
                        loss_func = WCCC_loss
                    elif loss == 'TRR':
                        loss_func = TRR()
                    else:
                        raise ValueError(f'The paramter loss should be , get {loss} instead')

                    print(f'Model {model_name} for factor 1 does not exist, training model ...')
                    model.to(device=device)
                    model.train()

                    if len(eval_dates) != 0:
                        eval_factor1 = th.from_numpy(eval_factor1)
                        eval_factor1 = eval_factor1.to(th.float32)
                        eval_factor1_dataset = Data(eval_factor1, train_Y)
                        eval_factor1_dataloader = DataLoader(eval_factor1_dataset, BATCH_SIZE, shuffle=False)
                        Best_model_factor1, _ = train(model, train_factor1_dataloader, loss_func, eval_factor1_dataloader, MAX_EPOCH)
                    else:
                        Best_model_factor1, _ = train(model, train_factor1_dataloader, loss_func, None, MAX_EPOCH)

                    joblib.dump(Best_model_factor1, f'{save_dir}/Model/{bench}/{model_name}_{loss}_factor1_{year_i}_{year_i+train_window-1}.m')

                predLst_factor1 = get_factor(Best_model_factor1, test_factor1_dataloader)

                if len(predLst_factor1.shape) == 1:
                    factor1_dict = {'id': test_id, 'date': test_tdate, 'return': test_Y}
                    factor1_dict[f'{model_name}_factor1'] = predLst_factor1
                elif len(predLst_factor1.shape) == 2:
                    factor1_dict = {'id': test_id, 'date': test_tdate, 'return': test_Y}
                    for i in range(predLst_factor1.shape[1]):
                        factor1_dict[f'{model_name}_factor1_{i+1}'] = predLst_factor1[:,i]

                factor1_df = pd.DataFrame(factor1_dict)

                if not os.path.exists(f'{save_dir}/Factor/{loss}/{bench}/Factor1/{model_name}'):
                    os.makedirs(f'{save_dir}/Factor/{loss}/{bench}/Factor1/{model_name}')

                # store the factor data
                factor1_tdate = np.sort(pd.unique(factor1_df['date']))
                with trange(len(factor1_tdate)) as date_bar:
                    for i in date_bar:
                        date_i = factor1_tdate[i]
                        date_bar.set_description(f'Saving data on date {date_i}')

                        factor1_df_i = factor1_df.loc[factor1_df['date'] == date_i]
                        factor1_df_i.to_csv(f'{save_dir}/Factor/{loss}/{bench}/Factor1/{model_name}/{date_i}.csv', index=False)

        if not factor2 is None:
            n_feature_factor2 = train_factor2.shape[-1]
            train_factor2, test_factor2 = np.nan_to_num(train_factor2, nan=0), np.nan_to_num(test_factor2, nan=0)
            train_factor2, test_factor2 = th.from_numpy(train_factor2), th.from_numpy(test_factor2)
            train_factor2, test_factor2 = train_factor2.to(th.float32), test_factor2.to(th.float32)
            train_factor2_dataset = Data(train_factor2, train_Y)
            test_factor2_dataset = Data(test_factor2, test_Y)
            train_factor2_dataloader = DataLoader(train_factor2_dataset, BATCH_SIZE, shuffle=False)
            test_factor2_dataloader = DataLoader(test_factor2_dataset, BATCH_SIZE, shuffle=False)

            # train the model one by one
            for model_name in modelLst:
                # train the model if and only if we haven't trained it before
                if os.path.exists(f'{save_dir}/Model/{bench}/{model_name}_{loss}_factor2_{year_i}_{year_i+train_window-1}.m'):
                    print(f'Model {model_name} for factor 2 already exists, loading model ...')
                    Best_model_factor2 = joblib.load(f'{save_dir}/Model/{bench}/{model_name}_{loss}_factor2_{year_i}_{year_i+train_window-1}.m')
                    Best_model_factor2.to(device=device)
                    Best_model_factor2.eval()

                else:
                    if model_name == 'LSTM':
                        model = LSTMModel(input_size=n_feature_factor2, output_size=n_factors)
                    elif model_name == 'GRU':
                        model = GRUModel(input_size=n_feature_factor2, output_size=n_factors)
                    elif model_name == 'ALSTM':
                        model = ALSTMModel(input_size=n_feature_factor2, output_size=n_factors)
                    elif model_name == 'TCN':
                        model = TCNModel(num_input=back_window, output_size=n_factors, num_feature=n_feature_factor2)
                    elif model_name == 'Transformer':
                        model = Transformer(input_size=n_feature_factor2, output_size=n_factors)
                    else:
                        raise ValueError(f'The parameter model should be LSTM/GRU/ALSTM/TCN/Transformer, get {model_name} instead')
                    
                    if loss == 'IC':
                        loss_func = IC_loss
                    elif loss == 'IC_penalty':
                        loss_func = IC_loss_penalty
                    elif loss == 'CCC':
                        loss_func = CCC_loss
                    elif loss == 'WCCC':
                        loss_func = WCCC_loss
                    elif loss == 'TRR':
                        loss_func = TRR()
                    elif loss == 'Closs_explained':
                        loss_func = Closs_explained()
                    else:
                        raise ValueError(f'The paramter loss should be , get {loss} instead')
                    
                    print(f'Model {model_name} for factor 2 does not exist, training model ...')
                    model.to(device=device)
                    model.train()

                    if len(eval_dates) != 0:
                        eval_factor2 = th.from_numpy(eval_factor2)
                        eval_factor2 = eval_factor2.to(th.float32)
                        eval_factor2_dataset = Data(eval_factor2, train_Y)
                        eval_factor2_dataloader = DataLoader(eval_factor2_dataset, BATCH_SIZE, shuffle=False)
                        Best_model_factor2, _ = train(model, train_factor2_dataloader, loss_func, eval_factor2_dataloader, MAX_EPOCH)
                    else:
                        Best_model_factor2, _ = train(model, train_factor2_dataloader, loss_func, None, MAX_EPOCH)

                    joblib.dump(Best_model_factor2, f'{save_dir}/Model/{bench}/{model_name}_{loss}_factor2_{year_i}_{year_i+train_window-1}.m')

                predLst_factor2 = get_factor(Best_model_factor2, test_factor2_dataloader)

                if len(predLst_factor2.shape) == 1:
                    factor2_dict = {'id': test_id, 'date': test_tdate, 'return': test_Y}
                    factor2_dict[f'{model_name}_factor2'] = predLst_factor2
                elif len(predLst_factor2.shape) == 2:
                    factor2_dict = {'id': test_id, 'date': test_tdate, 'return': test_Y}
                    for i in range(predLst_factor2.shape[1]):
                        factor2_dict[f'{model_name}_factor2_{i+1}'] = predLst_factor2[:,i]
                factor2_df = pd.DataFrame(factor2_dict)

                if not os.path.exists(f'{save_dir}/Factor/{loss}/{bench}/Factor2/{model_name}'):
                    os.makedirs(f'{save_dir}/Factor/{loss}/{bench}/Factor2/{model_name}')

                # store the factor data
                factor2_tdate = np.sort(pd.unique(factor2_df['date']))
                with trange(len(factor2_tdate)) as date_bar:
                    for i in date_bar:
                        date_i = factor2_tdate[i]
                        date_bar.set_description(f'Saving data on date {date_i}')

                        factor2_df_i = factor2_df.loc[factor2_df['date'] == date_i]
                        factor2_df_i.to_csv(f'{save_dir}/Factor/{loss}/{bench}/Factor2/{model_name}/{date_i}.csv', index=False)


# train and test data in the order of year
def train_and_test_rank(save_dir: str, trDays_dir: str, member_dir: str, modelLst: list, loss: str, start_year: int, end_year: int, train_window: int, 
                   eval_window: int, test_window: int, n_factors: int, back_window: int, bench: str, T: int, stock: str, factor1: str, factor2: str,
                   n_stocks: int, gap: bool, save_npy: bool, member: str):

    # create folder to save model and rolling data
    if not os.path.exists(f'{save_dir}/Model_Rank/{bench}'):
        os.makedirs(f'{save_dir}/Model_Rank/{bench}')

    if not os.path.exists(f'{save_dir}/Rolling_Rank/{bench}'):
        os.makedirs(f'{save_dir}/Rolling_Rank/{bench}')

    # load the trading date data within the corresponding time interval
    timeLst = read_pkl5_data(trDays_dir)
    timeLst = timeLst.index.to_numpy()
    timeLst = timeLst[(int(f'{start_year-1}0101') <= timeLst) & (timeLst <= int(f'{end_year}1231'))]

    # # get the stock return data
    # return_df = pd.DataFrame()

    # with trange(len(timeLst)) as date_bar:
    #     for i in date_bar:
    #         date_i = timeLst[i]
    #         date_bar.set_description(f'Loading data on date {date_i}')

    #         if gap:
    #             try:
    #                 return_i = pd.read_csv(f'{save_dir}/Data/Return/allAshare/T+1_T+{T+1}/{date_i}.csv')
    #             except:
    #                 print(f'No such file: {save_dir}/Data/Return/allAshare/T+1_T+{T+1}/{date_i}.csv')
    #                 continue
    #         else:
    #             try:
    #                 return_i = pd.read_csv(f'{save_dir}/Data/Return/allAshare/T_T+{T}/{date_i}.csv')
    #             except:
    #                 print(f'No such file: {save_dir}/Data/Return/allAshare/T_T+{T}/{date_i}.csv')

    #         return_df = return_df.append(return_i)

    # iterately train and test the model
    for year_i in range(start_year, end_year-train_window-eval_window-test_window+2):

        print(f'''processing data with training period {year_i}-{year_i+train_window-1}
        evaluation period {year_i+train_window}-{year_i+train_window+eval_window-1}
        and testing period {year_i+train_window+eval_window}-{year_i+train_window+eval_window+test_window-1}''')

        # get the dates for corresponding data sets
        train_dates = timeLst[(int(f'{year_i}0101') <= timeLst) &
             (timeLst <= int(f'{year_i+train_window-1}1231'))]
        eval_dates = timeLst[(int(f'{year_i+train_window}0101') <= timeLst) &
             (timeLst <= int(f'{year_i+train_window+eval_window-1}1231'))]
        test_dates = timeLst[(int(f'{year_i+train_window+eval_window}0101') <= timeLst) &
             (timeLst <= int(f'{year_i+train_window+eval_window+test_window-1}1231'))]
        
        if bench != 'allAshare' and member == 'topN':
            member_df = read_member(member_dir, train_dates[0], test_dates[-1])
            member_df['cnt'] = 1
            member_df = member_df.groupby('id')['cnt'].sum()
            member_df = member_df.to_frame()
            member_df.reset_index(drop=False, inplace=True)
        elif bench != 'allAshare' and member == 'bench':
            member_df = read_member(member_dir, train_dates[0], test_dates[-1])
        else:
            member_df = None

        # load training and testing datasets
        if save_npy:
            if os.path.exists(f'{save_dir}/Rolling_Rank/{bench}/{train_dates[0]}_{train_dates[-1]}_stock.npy'):
                # train_stock, train_factor1, train_factor2, train_Y, train_id, train_tdate = np.load(f'{save_dir}/Rolling/{bench}/{train_dates[0]}_{train_dates[-1]}_data.npy', allow_pickle=True)
                train_stock = np.load(f'{save_dir}/Rolling_Rank/{bench}/{train_dates[0]}_{train_dates[-1]}_stock.npy', allow_pickle=True)
                train_factor1 = np.load(f'{save_dir}/Rolling_Rank/{bench}/{train_dates[0]}_{train_dates[-1]}_factor1.npy', allow_pickle=True)
                train_factor2 = np.load(f'{save_dir}/Rolling_Rank/{bench}/{train_dates[0]}_{train_dates[-1]}_factor2.npy', allow_pickle=True)
                train_Y = np.load(f'{save_dir}/Rolling_Rank/{bench}/{train_dates[0]}_{train_dates[-1]}_label.npy', allow_pickle=True)
                train_id = np.load(f'{save_dir}/Rolling_Rank/{bench}/{train_dates[0]}_{train_dates[-1]}_id.npy', allow_pickle=True)
                train_tdate = np.load(f'{save_dir}/Rolling_Rank/{bench}/{train_dates[0]}_{train_dates[-1]}_date.npy', allow_pickle=True)
            else:
                train_stock, train_factor1, train_factor2, train_Y, train_id, train_tdate = load_dataset_rank(save_dir, train_dates, return_df, member_df, stock, factor1, factor2, back_window, n_stocks, member)
                # train_data = np.empty(6, dtype=object)
                # train_data[:] = [train_stock, train_factor1, train_factor2, train_Y, train_id, train_tdate]
                # np.save(f'{save_dir}/Rolling/{bench}/{train_dates[0]}_{train_dates[-1]}_data.npy', train_data, allow_pickle=True)
                np.save(f'{save_dir}/Rolling_Rank/{bench}/{train_dates[0]}_{train_dates[-1]}_stock.npy', train_stock, allow_pickle=True)
                np.save(f'{save_dir}/Rolling_Rank/{bench}/{train_dates[0]}_{train_dates[-1]}_factor1.npy', train_factor1, allow_pickle=True)
                np.save(f'{save_dir}/Rolling_Rank/{bench}/{train_dates[0]}_{train_dates[-1]}_factor2.npy', train_factor2, allow_pickle=True)
                np.save(f'{save_dir}/Rolling_Rank/{bench}/{train_dates[0]}_{train_dates[-1]}_label.npy', train_Y, allow_pickle=True)
                np.save(f'{save_dir}/Rolling_Rank/{bench}/{train_dates[0]}_{train_dates[-1]}_id.npy', train_id, allow_pickle=True)
                np.save(f'{save_dir}/Rolling_Rank/{bench}/{train_dates[0]}_{train_dates[-1]}_date.npy', train_tdate, allow_pickle=True)
        else:
            train_stock, train_factor1, train_factor2, train_Y, train_id, train_tdate = load_dataset_rank(save_dir, train_dates, return_df, member_df, stock, factor1, factor2, back_window, n_stocks, member)

        train_return = pd.DataFrame({'id': train_id, 'date': train_tdate, 'return': train_Y})
        train_return.set_index(['id', 'date'], inplace=True)
        train_return = train_return.unstack().values
        train_return = np.nan_to_num(train_return, nan=0)
        train_return = np.transpose(train_return, (1, 0))

        if save_npy:
            if os.path.exists(f'{save_dir}/Rolling_Rank/{bench}/{test_dates[0]}_{test_dates[-1]}_stock.npy'):
                # test_stock, test_factor1, test_factor2, test_Y, test_id, test_tdate = np.load(f'{save_dir}/Rolling/{bench}/{test_dates[0]}_{test_dates[-1]}_data.npy', allow_pickle=True)
                test_stock = np.load(f'{save_dir}/Rolling_Rank/{bench}/{test_dates[0]}_{test_dates[-1]}_stock.npy', allow_pickle=True)
                test_factor1 = np.load(f'{save_dir}/Rolling_Rank/{bench}/{test_dates[0]}_{test_dates[-1]}_factor1.npy', allow_pickle=True)
                test_factor2 = np.load(f'{save_dir}/Rolling_Rank/{bench}/{test_dates[0]}_{test_dates[-1]}_factor2.npy', allow_pickle=True)
                test_Y = np.load(f'{save_dir}/Rolling_Rank/{bench}/{test_dates[0]}_{test_dates[-1]}_label.npy', allow_pickle=True)
                test_id = np.load(f'{save_dir}/Rolling_Rank/{bench}/{test_dates[0]}_{test_dates[-1]}_id.npy', allow_pickle=True)
                test_tdate = np.load(f'{save_dir}/Rolling_Rank/{bench}/{test_dates[0]}_{test_dates[-1]}_date.npy', allow_pickle=True)
            else:
                test_stock, test_factor1, test_factor2, test_Y, test_id, test_tdate = load_dataset_rank(save_dir, test_dates, return_df, member_df, stock, factor1, factor2, back_window, n_stocks, member)
                # test_data = np.empty(6, dtype=object)
                # test_data[:] = [test_stock, test_factor1, test_factor2, test_Y, test_id, test_tdate]
                # np.save(f'{save_dir}/Rolling/{bench}/{test_dates[0]}_{test_dates[-1]}_data.npy', test_data, allow_pickle=True)
                np.save(f'{save_dir}/Rolling_Rank/{bench}/{test_dates[0]}_{test_dates[-1]}_stock.npy', test_stock, allow_pickle=True)
                np.save(f'{save_dir}/Rolling_Rank/{bench}/{test_dates[0]}_{test_dates[-1]}_factor1.npy', test_factor1, allow_pickle=True)
                np.save(f'{save_dir}/Rolling_Rank/{bench}/{test_dates[0]}_{test_dates[-1]}_factor2.npy', test_factor2, allow_pickle=True)
                np.save(f'{save_dir}/Rolling_Rank/{bench}/{test_dates[0]}_{test_dates[-1]}_label.npy', test_Y, allow_pickle=True)
                np.save(f'{save_dir}/Rolling_Rank/{bench}/{test_dates[0]}_{test_dates[-1]}_id.npy', test_id, allow_pickle=True)
                np.save(f'{save_dir}/Rolling_Rank/{bench}/{test_dates[0]}_{test_dates[-1]}_date.npy', test_tdate, allow_pickle=True)
        else:
            test_stock, test_factor1, test_factor2, test_Y, test_id, test_tdate = load_dataset_rank(save_dir, test_dates, return_df, member_df, stock, factor1, factor2, back_window, n_stocks, member)

        test_return = pd.DataFrame({'id': test_id, 'date': test_tdate, 'return': test_Y})
        test_return.set_index(['id', 'date'], inplace=True)
        test_return = test_return.unstack().values
        test_return = np.nan_to_num(test_return, nan=0)
        test_return = np.transpose(test_return, (1, 0))

        # load the evaluation data set if it exists
        if len(eval_dates) != 0:
            if save_npy:
                if os.path.exists(f'{save_dir}/Rolling_Rank/{bench}/{eval_dates[0]}_{eval_dates[-1]}_stock.npy'):
                    # eval_stock, eval_factor1, eval_factor2, eval_Y, eval_id, eval_tdate = np.load(f'{save_dir}/Rolling/{bench}/{eval_dates[0]}_{eval_dates[-1]}_data.npy', allow_pickle=True)
                    eval_stock = np.load(f'{save_dir}/Rolling_Rank/{bench}/{eval_dates[0]}_{eval_dates[-1]}_stock.npy', allow_pickle=True)
                    eval_factor1 = np.load(f'{save_dir}/Rolling_Rank/{bench}/{eval_dates[0]}_{eval_dates[-1]}_factor1.npy', allow_pickle=True)
                    eval_factor2 = np.load(f'{save_dir}/Rolling_Rank/{bench}/{eval_dates[0]}_{eval_dates[-1]}_factor2.npy', allow_pickle=True)
                    eval_Y = np.load(f'{save_dir}/Rolling_Rank/{bench}/{eval_dates[0]}_{eval_dates[-1]}_label.npy', allow_pickle=True)
                    eval_id = np.load(f'{save_dir}/Rolling_Rank/{bench}/{eval_dates[0]}_{eval_dates[-1]}_id.npy', allow_pickle=True)
                    eval_tdate = np.load(f'{save_dir}/Rolling_Rank/{bench}/{eval_dates[0]}_{eval_dates[-1]}_date.npy', allow_pickle=True)
                else:
                    eval_stock, eval_factor1, eval_factor2, eval_Y, eval_id, eval_tdate = load_dataset_rank(save_dir, eval_dates, return_df, member_df, stock, factor1, factor2, back_window, n_stocks, member)
                    # eval_data = np.empty(6, dtype=object)
                    # eval_data[:] = [eval_stock, eval_factor1, eval_factor2, eval_Y, eval_id, eval_tdate]
                    # np.save(f'{save_dir}/Rolling/{bench}/{eval_dates[0]}_{eval_dates[-1]}_data.npy', eval_data, allow_pickle=True)
                    np.save(f'{save_dir}/Rolling_Rank/{bench}/{eval_dates[0]}_{eval_dates[-1]}_stock.npy', eval_stock, allow_pickle=True)
                    np.save(f'{save_dir}/Rolling_Rank/{bench}/{eval_dates[0]}_{eval_dates[-1]}_factor1.npy', eval_factor1, allow_pickle=True)
                    np.save(f'{save_dir}/Rolling_Rank/{bench}/{eval_dates[0]}_{eval_dates[-1]}_factor2.npy', eval_factor2, allow_pickle=True)
                    np.save(f'{save_dir}/Rolling_Rank/{bench}/{eval_dates[0]}_{eval_dates[-1]}_label.npy', eval_Y, allow_pickle=True)
                    np.save(f'{save_dir}/Rolling_Rank/{bench}/{eval_dates[0]}_{eval_dates[-1]}_id.npy', eval_id, allow_pickle=True)
                    np.save(f'{save_dir}/Rolling_Rank/{bench}/{eval_dates[0]}_{eval_dates[-1]}_date.npy', eval_tdate, allow_pickle=True)
            else:
                eval_stock, eval_factor1, eval_factor2, eval_Y, eval_id, eval_tdate = load_dataset_rank(save_dir, eval_dates, return_df, member_df, stock, factor1, factor2, back_window, n_stocks, member)

            eval_return = pd.DataFrame({'id': eval_id, 'date': eval_tdate, 'return': eval_Y})
            eval_return.set_index(['id', 'date'], inplace=True)
            eval_return = eval_return.unstack().values
            eval_return = np.nan_to_num(eval_return, nan=0)
            eval_return = np.transpose(eval_return, (1, 0))
        
        # create data loaders for training and testing datasets
        if not stock is None:
            n_feature_stock = train_stock.shape[-1]
            train_stock, test_stock = np.nan_to_num(train_stock, nan=0), np.nan_to_num(test_stock, nan=0)
            train_stock, test_stock = th.from_numpy(train_stock), th.from_numpy(test_stock)
            train_stock, test_stock = np.transpose(train_stock, (1, 0, 2)), np.transpose(test_stock, (1, 0, 2))
            train_stock, test_stock = train_stock.to(th.float32), test_stock.to(th.float32)
            train_stock_dataset = Data(train_stock, train_return)
            test_stock_dataset = Data(test_stock, test_return)
            train_stock_dataloader = DataLoader(train_stock_dataset, BATCH_SIZE, shuffle=False)
            test_stock_dataloader = DataLoader(test_stock_dataset, BATCH_SIZE, shuffle=False)

            # train the model one by one
            for model_name in modelLst:
                # train the model if and only if we haven't trained it before
                if os.path.exists(f'{save_dir}/Model_Rank/{bench}/{model_name}_{loss}_stock_{year_i}_{year_i+train_window-1}.m'):
                    print(f'Model {model_name} for stock already exists, loading model ...')
                    Best_model_stock = joblib.load(f'{save_dir}/Model_Rank/{bench}/{model_name}_{loss}_stock_{year_i}_{year_i+train_window-1}.m')
                    Best_model_stock.to(device=device)
                    Best_model_stock.eval()

                else:
                    if model_name == 'LSTM':
                        model = LSTMRankModel(input_size=n_feature_stock, output_size=n_factors)
                    elif model_name == 'GRU':
                        model = GRURankModel(input_size=n_feature_stock, output_size=n_factors)
                    elif model_name == 'ALSTM':
                        model = ALSTMRankModel(input_size=n_feature_stock, output_size=n_factors)
                    elif model_name == 'TCN':
                        model = TCNRankModel(num_input=back_window, output_size=n_factors, num_feature=n_feature_stock)
                    elif model_name == 'Transformer':
                        model = TransformerRank(input_size=n_feature_stock, output_size=n_factors)
                    else:
                        raise ValueError(f'The parameter model should be LSTM/GRU/ALSTM/TCN/Transformer, get {model_name} instead')
                    
                    if loss == 'Closs':
                        loss_func = Closs()
                    elif loss == 'Closs_explained':
                        loss_func = Closs_explained()
                    elif loss == 'Closs_sigmoid':
                        loss_func = Closs_sigmoid()
                    elif loss == 'Lloss':
                        loss_func = Lloss()
                    elif loss == 'ListMLE':
                        loss_func = ListMLE()
                    elif loss == 'ListNet':
                        loss_func = ListNet()
                    elif loss == 'Closs_penalty':
                        loss_func = Closs_penalty()
                    elif loss == 'Closs_explained_penalty':
                        loss_func = Closs_explained_penalty()
                    elif loss == 'Closs_sigmoid_penalty':
                        loss_func = Closs_sigmoid_penalty()
                    elif loss == 'Lloss_penalty':
                        loss_func = Lloss_penalty()
                    elif loss == 'ListMLE_penalty':
                        loss_func = ListMLE_penalty()
                    elif loss == 'ListNet_penalty':
                        loss_func = ListNet_penalty()
                    else:
                        raise ValueError(f'The parameter loss should be Closs/Closs_explained/Closs_igmoid/Lloss/ListMLE/ListNet, get {loss} instead')
                    
                    print(f'Model {model_name} for stock does not exist, training model ...')
                    model.to(device=device)
                    model.train()

                    if len(eval_dates) != 0:
                        eval_stock = np.nan_to_num(eval_stock, nan=0)
                        eval_stock = th.from_numpy(eval_stock)
                        eval_stock = eval_stock.to(th.float32)
                        eval_stock_dataset = Data(eval_stock, eval_return)
                        eval_stock_dataloader = DataLoader(eval_stock_dataset, BATCH_SIZE, shuffle=False)
                        if loss in ['Closs', 'Closs_explained', 'Closs_sigmoid', 'Lloss', 'Closs_penalty', 'Closs_explained_penalty', 'Closs_sigmoid_penalty', 'Lloss_penalty']:
                            Best_model_stock, _ = train_rank(model, train_stock_dataloader, loss_func, int(n_stocks/10), eval_stock_dataloader, MAX_EPOCH)
                        else:
                            Best_model_stock, _ = train(model, train_stock_dataloader, loss_func, eval_stock_dataloader, MAX_EPOCH)
                    else:
                        if loss in ['Closs', 'Closs_explained', 'Closs_sigmoid', 'Lloss', 'Closs_penalty', 'Closs_explained_penalty', 'Closs_sigmoid_penalty', 'Lloss_penalty']:
                            Best_model_stock, _ = train_rank(model, train_stock_dataloader, loss_func, int(n_stocks/10), None, MAX_EPOCH)
                        else:
                            Best_model_stock, _ = train(model, train_stock_dataloader, loss_func, None, MAX_EPOCH)

                    joblib.dump(Best_model_stock, f'{save_dir}/Model_Rank/{bench}/{model_name}_{loss}_stock_{year_i}_{year_i+train_window-1}.m')

                predLst_stock = get_factor(Best_model_stock, test_stock_dataloader)
                if len(predLst_stock.shape) == 2:
                    predLst_stock = np.transpose(predLst_stock, (1, 0))
                    stock_dict = {'id': test_id, 'date': test_tdate, 'return': test_Y}
                    stock_df = pd.DataFrame(stock_dict)
                    return_df = pd.DataFrame(stock_dict)
                    stock_df.set_index(['id', 'date'], inplace=True)
                    stock_df = stock_df.unstack()
                    stock_df.loc[:,:] = predLst_stock
                    stock_df = stock_df.stack()
                    stock_df.reset_index(drop=False, inplace=True)
                    stock_df.rename(columns={'return': f'{model_name}_stock'}, inplace=True)
                    stock_df = pd.merge(stock_df, return_df, on=['id', 'date'])
                elif len(predLst_stock.shape) == 3:
                    stockDFLst = []
                    for i in range(predLst_stock.shape[-1]):
                        temp_stock = np.transpose(predLst_stock[:,:,i], (1, 0))
                        stock_dict = {'id': test_id, 'date': test_tdate, 'return': test_Y}
                        stock_df = pd.DataFrame(stock_dict)
                        stock_df.set_index(['id', 'date'], inplace=True)
                        stock_df = stock_df.unstack()
                        stock_df.loc[:,:] = temp_stock
                        stock_df = stock_df.stack()
                        stock_df.reset_index(drop=False, inplace=True)
                        stock_df.rename(columns={'return': f'{model_name}_stock_{i+1}'}, inplace=True)
                        stockDFLst.append(stock_df)
                    return_df = pd.DataFrame(stock_dict)
                    stock_df = reduce(lambda df1, df2: pd.merge(df1, df2, on=['id', 'date']), stockDFLst)
                    stock_df = pd.merge(stock_df, return_df, on=['id', 'date'])

                if not os.path.exists(f'{save_dir}/Factor_Rank/{loss}/{bench}/Stock/{model_name}'):
                    os.makedirs(f'{save_dir}/Factor_Rank/{loss}/{bench}/Stock/{model_name}')

                # store the factor data
                stock_tdate = np.sort(pd.unique(stock_df['date']))
                with trange(len(stock_tdate)) as date_bar:
                    for i in date_bar:
                        date_i = stock_tdate[i]
                        date_bar.set_description(f'Saving data on date {date_i}')

                        stock_df_i = stock_df.loc[stock_df['date'] == date_i]
                        stock_df_i.to_csv(f'{save_dir}/Factor_Rank/{loss}/{bench}/Stock/{model_name}/{date_i}.csv', index=False)

        if not factor1 is None:
            n_feature_factor1 = train_factor1.shape[-1]
            train_factor1, test_factor1 = np.nan_to_num(train_factor1, nan=0), np.nan_to_num(test_factor1, nan=0)
            train_factor1, test_factor1 = np.transpose(train_factor1, (1, 0, 2)), np.transpose(test_factor1, (1, 0, 2))
            train_factor1, test_factor1 = th.from_numpy(train_factor1), th.from_numpy(test_factor1)
            train_factor1, test_factor1 = train_factor1.to(th.float32), test_factor1.to(th.float32)
            train_factor1_dataset = Data(train_factor1, train_return)
            test_factor1_dataset = Data(test_factor1, test_return)
            train_factor1_dataloader = DataLoader(train_factor1_dataset, BATCH_SIZE, shuffle=False)
            test_factor1_dataloader = DataLoader(test_factor1_dataset, BATCH_SIZE, shuffle=False)

            # train the model one by one
            for model_name in modelLst:
                # train the model if and only if we haven't trained it before
                if os.path.exists(f'{save_dir}/Model_Rank/{bench}/{model_name}_{loss}_factor1_{year_i}_{year_i+train_window-1}.m'):
                    print(f'Model {model_name} for factor 1 already exists, loading model ...')
                    Best_model_factor1 = joblib.load(f'{save_dir}/Model_Rank/{bench}/{model_name}_{loss}_factor1_{year_i}_{year_i+train_window-1}.m')
                    Best_model_factor1.to(device=device)
                    Best_model_factor1.eval()

                else:
                    if model_name == 'LSTM':
                        model = LSTMRankModel(input_size=n_feature_factor1, output_size=n_factors)
                    elif model_name == 'GRU':
                        model = GRURankModel(input_size=n_feature_factor1, output_size=n_factors)
                    elif model_name == 'ALSTM':
                        model = ALSTMRankModel(input_size=n_feature_factor1, output_size=n_factors)
                    elif model_name == 'TCN':
                        model = TCNRankModel(num_input=back_window, output_size=n_factors, num_feature=n_feature_factor1)
                    elif model_name == 'Transformer':
                        model = TransformerRank(input_size=n_feature_factor1, output_size=n_factors)
                    else:
                        raise ValueError(f'The parameter model should be LSTM/GRU/ALSTM/TCN/Transformer, get {model_name} instead')
                    
                    if loss == 'Closs':
                        loss_func = Closs()
                    elif loss == 'Closs_explained':
                        loss_func = Closs_explained()
                    elif loss == 'Closs_sigmoid':
                        loss_func = Closs_sigmoid()
                    elif loss == 'Lloss':
                        loss_func = Lloss()
                    elif loss == 'ListMLE':
                        loss_func = ListMLE()
                    elif loss == 'ListNet':
                        loss_func = ListNet()
                    elif loss == 'Closs_penalty':
                        loss_func = Closs_penalty()
                    elif loss == 'Closs_explained_penalty':
                        loss_func = Closs_explained_penalty()
                    elif loss == 'Closs_sigmoid_penalty':
                        loss_func = Closs_sigmoid_penalty()
                    elif loss == 'Lloss_penalty':
                        loss_func = Lloss_penalty()
                    elif loss == 'ListMLE_penalty':
                        loss_func = ListMLE_penalty()
                    elif loss == 'ListNet_penalty':
                        loss_func = ListNet_penalty()
                    else:
                        raise ValueError(f'The parameter loss should be Closs/Closs_explained/Closs_igmoid/Lloss/ListMLE/ListNet, get {loss} instead')

                    print(f'Model {model_name} for factor 1 does not exist, training model ...')
                    model.to(device=device)
                    model.train()

                    if len(eval_dates) != 0:
                        eval_factor1 = np.nan_to_num(eval_factor1, nan=0)
                        eval_factor1 = th.from_numpy(eval_factor1)
                        eval_factor1 = eval_factor1.to(th.float32)
                        eval_factor1_dataset = Data(eval_factor1, eval_return)
                        eval_factor1_dataloader = DataLoader(eval_factor1_dataset, BATCH_SIZE, shuffle=False)
                        if loss in ['Closs', 'Closs_explained', 'Closs_sigmoid', 'Lloss', 'Closs_penalty', 'Closs_explained_penalty', 'Closs_sigmoid_penalty', 'Lloss_penalty']:
                            Best_model_factor1, _ = train_rank(model, train_factor1_dataloader, loss_func, int(n_stocks/10), eval_factor1_dataloader, MAX_EPOCH)
                        else:
                            Best_model_factor1, _ = train(model, train_factor1_dataloader, loss_func, eval_factor1_dataloader, MAX_EPOCH)
                    else:
                        if loss in ['Closs', 'Closs_explained', 'Closs_sigmoid', 'Lloss', 'Closs_penalty', 'Closs_explained_penalty', 'Closs_sigmoid_penalty', 'Lloss_penalty']:
                            Best_model_factor1, _ = train_rank(model, train_factor1_dataloader, loss_func, int(n_stocks/10), None, MAX_EPOCH)
                        else:
                            Best_model_factor1, _ = train(model, train_factor1_dataloader, loss_func, None, MAX_EPOCH)

                    joblib.dump(Best_model_factor1, f'{save_dir}/Model_Rank/{bench}/{model_name}_{loss}_factor1_{year_i}_{year_i+train_window-1}.m')

                predLst_factor1 = get_factor(Best_model_factor1, test_factor1_dataloader)
                if len(predLst_factor1.shape) == 2:
                    predLst_factor1 = np.transpose(predLst_factor1, (1, 0))
                    factor1_dict = {'id': test_id, 'date': test_tdate, 'return': test_Y}
                    factor1_df = pd.DataFrame(factor1_dict)
                    return_df = pd.DataFrame(factor1_dict)
                    factor1_df.set_index(['id', 'date'], inplace=True)
                    factor1_df = factor1_df.unstack()
                    factor1_df.loc[:,:] = predLst_factor1
                    factor1_df = factor1_df.stack()
                    factor1_df.reset_index(drop=False, inplace=True)
                    factor1_df.rename(columns={'return': f'{model_name}_factor1'}, inplace=True)
                    factor1_df = pd.merge(factor1_df, return_df, on=['id', 'date'])
                elif len(predLst_factor1.shape) == 3:
                    factor1DFLst = []
                    for i in range(predLst_factor1.shape[-1]):
                        temp_factor1 = np.transpose(predLst_factor1[:,:,i], (1, 0))
                        factor1_dict = {'id': test_id, 'date': test_tdate, 'return': test_Y}
                        factor1_df = pd.DataFrame(factor1_dict)
                        factor1_df.set_index(['id', 'date'], inplace=True)
                        factor1_df = factor1_df.unstack()
                        factor1_df.loc[:,:] = temp_factor1
                        factor1_df = factor1_df.stack()
                        factor1_df.reset_index(drop=False, inplace=True)
                        factor1_df.rename(columns={'return': f'{model_name}_factor1_{i+1}'}, inplace=True)
                        factor1DFLst.append(factor1_df)
                    return_df = pd.DataFrame(factor1_dict)
                    factor1_df = reduce(lambda df1, df2: pd.merge(df1, df2, on=['id', 'date']), factor1DFLst)
                    factor1_df = pd.merge(factor1_df, return_df, on=['id', 'date'])

                if not os.path.exists(f'{save_dir}/Factor_Rank/{loss}/{bench}/Factor1/{model_name}'):
                    os.makedirs(f'{save_dir}/Factor_Rank/{loss}/{bench}/Factor1/{model_name}')

                # store the factor data
                factor1_tdate = np.sort(pd.unique(factor1_df['date']))
                with trange(len(factor1_tdate)) as date_bar:
                    for i in date_bar:
                        date_i = factor1_tdate[i]
                        date_bar.set_description(f'Saving data on date {date_i}')

                        factor1_df_i = factor1_df.loc[factor1_df['date'] == date_i]
                        factor1_df_i.to_csv(f'{save_dir}/Factor_Rank/{loss}/{bench}/Factor1/{model_name}/{date_i}.csv', index=False)

        if not factor2 is None:
            n_feature_factor2 = train_factor2.shape[-1]
            train_factor2, test_factor2 = np.nan_to_num(train_factor2, nan=0), np.nan_to_num(test_factor2, nan=0)
            train_factor2, test_factor2 = np.transpose(train_factor2, (1, 0, 2)), np.transpose(test_factor2, (1, 0, 2))
            train_factor2, test_factor2 = th.from_numpy(train_factor2), th.from_numpy(test_factor2)
            train_factor2, test_factor2 = train_factor2.to(th.float32), test_factor2.to(th.float32)
            train_factor2_dataset = Data(train_factor2, train_return)
            test_factor2_dataset = Data(test_factor2, test_return)
            train_factor2_dataloader = DataLoader(train_factor2_dataset, BATCH_SIZE, shuffle=False)
            test_factor2_dataloader = DataLoader(test_factor2_dataset, BATCH_SIZE, shuffle=False)

            # train the model one by one
            for model_name in modelLst:
                # train the model if and only if we haven't trained it before
                if os.path.exists(f'{save_dir}/Model_Rank/{bench}/{model_name}_{loss}_factor2_{year_i}_{year_i+train_window-1}.m'):
                    print(f'Model {model_name} for factor 2 already exists, loading model ...')
                    Best_model_factor2 = joblib.load(f'{save_dir}/Model_Rank/{bench}/{model_name}_{loss}_factor2_{year_i}_{year_i+train_window-1}.m')
                    Best_model_factor2.to(device=device)
                    Best_model_factor2.eval()

                else:
                    if model_name == 'LSTM':
                        model = LSTMRankModel(input_size=n_feature_factor2, output_size=n_factors)
                    elif model_name == 'GRU':
                        model = GRURankModel(input_size=n_feature_factor2, output_size=n_factors)
                    elif model_name == 'ALSTM':
                        model = ALSTMRankModel(input_size=n_feature_factor2, output_size=n_factors)
                    elif model_name == 'TCN':
                        model = TCNRankModel(num_input=back_window, output_size=n_factors, num_feature=n_feature_factor2)
                    elif model_name == 'Transformer':
                        model = TransformerRank(input_size=n_feature_factor2, output_size=n_factors)
                    else:
                        raise ValueError(f'The parameter model should be LSTM/GRU/ALSTM/TCN/Transformer, get {model_name} instead')
                    
                    if loss == 'Closs':
                        loss_func = Closs()
                    elif loss == 'Closs_explained':
                        loss_func = Closs_explained()
                    elif loss == 'Closs_sigmoid':
                        loss_func = Closs_sigmoid()
                    elif loss == 'Lloss':
                        loss_func = Lloss()
                    elif loss == 'ListMLE':
                        loss_func = ListMLE()
                    elif loss == 'ListNet':
                        loss_func = ListNet()
                    elif loss == 'Closs_penalty':
                        loss_func = Closs_penalty()
                    elif loss == 'Closs_explained_penalty':
                        loss_func = Closs_explained_penalty()
                    elif loss == 'Closs_sigmoid_penalty':
                        loss_func = Closs_sigmoid_penalty()
                    elif loss == 'Lloss_penalty':
                        loss_func = Lloss_penalty()
                    elif loss == 'ListMLE_penalty':
                        loss_func = ListMLE_penalty()
                    elif loss == 'ListNet_penalty':
                        loss_func = ListNet_penalty()
                    else:
                        raise ValueError(f'The parameter loss should be Closs/Closs_explained/Closs_igmoid/Lloss/ListMLE/ListNet, get {loss} instead')
                    
                    print(f'Model {model_name} for factor 2 does not exist, training model ...')
                    model.to(device=device)
                    model.train()

                    if len(eval_dates) != 0:
                        eval_factor2 = np.nan_to_num(eval_factor2, nan=0)
                        eval_factor2 = th.from_numpy(eval_factor2)
                        eval_factor2 = eval_factor2.to(th.float32)
                        eval_factor2_dataset = Data(eval_factor2, eval_return)
                        eval_factor2_dataloader = DataLoader(eval_factor2_dataset, BATCH_SIZE, shuffle=False)
                        if loss in ['Closs', 'Closs_explained', 'Closs_sigmoid', 'Lloss', 'Closs_penalty', 'Closs_explained_penalty', 'Closs_sigmoid_penalty', 'Lloss_penalty']:
                            Best_model_factor2, _ = train_rank(model, train_factor2_dataloader, loss_func, int(n_stocks/10), eval_factor2_dataloader, MAX_EPOCH)
                        else:
                            Best_model_factor2, _ = train(model, train_factor2_dataloader, loss_func, eval_factor2_dataloader, MAX_EPOCH)
                    else:
                        if loss in ['Closs', 'Closs_explained', 'Closs_sigmoid', 'Lloss', 'Closs_penalty', 'Closs_explained_penalty', 'Closs_sigmoid_penalty', 'Lloss_penalty']:
                            Best_model_factor2, _ = train_rank(model, train_factor2_dataloader, loss_func, int(n_stocks/10), None, MAX_EPOCH)
                        else:
                            Best_model_factor2, _ = train(model, train_factor2_dataloader, loss_func, None, MAX_EPOCH)

                    joblib.dump(Best_model_factor2, f'{save_dir}/Model_Rank/{bench}/{model_name}_{loss}_factor2_{year_i}_{year_i+train_window-1}.m')

                predLst_factor2 = get_factor(Best_model_factor2, test_factor2_dataloader)
                if len(predLst_factor2.shape) == 2:
                    predLst_factor2 = np.transpose(predLst_factor2, (1, 0))
                    factor2_dict = {'id': test_id, 'date': test_tdate, 'return': test_Y}
                    factor2_df = pd.DataFrame(factor2_dict)
                    return_df = pd.DataFrame(factor2_dict)
                    factor2_df.set_index(['id', 'date'], inplace=True)
                    factor2_df = factor2_df.unstack()
                    factor2_df.loc[:,:] = predLst_factor2
                    factor2_df = factor2_df.stack()
                    factor2_df.reset_index(drop=False, inplace=True)
                    factor2_df.rename(columns={'return': f'{model_name}_factor2'}, inplace=True)
                    factor2_df = pd.merge(factor2_df, return_df, on=['id', 'date'])
                elif len(predLst_factor2.shape) == 3:
                    factor2DFLst = []
                    for i in range(predLst_factor2.shape[-1]):
                        temp_factor2 = np.transpose(predLst_factor2[:,:,i], (1, 0))
                        factor2_dict = {'id': test_id, 'date': test_tdate, 'return': test_Y}
                        factor2_df = pd.DataFrame(factor2_dict)
                        factor2_df.set_index(['id', 'date'], inplace=True)
                        factor2_df = factor2_df.unstack()
                        factor2_df.loc[:,:] = temp_factor2
                        factor2_df = factor2_df.stack()
                        factor2_df.reset_index(drop=False, inplace=True)
                        factor2_df.rename(columns={'return': f'{model_name}_factor2_{i+1}'}, inplace=True)
                        factor2DFLst.append(factor2_df)
                    return_df = pd.DataFrame(factor2_dict)
                    factor2_df = reduce(lambda df1, df2: pd.merge(df1, df2, on=['id', 'date']), factor2DFLst)
                    factor2_df = pd.merge(factor2_df, return_df, on=['id', 'date'])

                if not os.path.exists(f'{save_dir}/Factor_Rank/{loss}/{bench}/Factor2/{model_name}'):
                    os.makedirs(f'{save_dir}/Factor_Rank/{loss}/{bench}/Factor2/{model_name}')

                # store the factor data
                factor2_tdate = np.sort(pd.unique(factor2_df['date']))
                with trange(len(factor2_tdate)) as date_bar:
                    for i in date_bar:
                        date_i = factor2_tdate[i]
                        date_bar.set_description(f'Saving data on date {date_i}')

                        factor2_df_i = factor2_df.loc[factor2_df['date'] == date_i]
                        factor2_df_i.to_csv(f'{save_dir}/Factor_Rank/{loss}/{bench}/Factor2/{model_name}/{date_i}.csv', index=False)

# evaluate the performance of the nn model
def evaluate(model, dataloader, criterion):

    # enumerate mini batches
    print('Evaluating ...')
    test_data_size = len(dataloader)
    test_dataiter = iter(dataloader)
    model.eval()

    # initailize the loss
    Loss = 0

    # set the bar to check the progress
    with trange(test_data_size) as test_bar:
        for i in test_bar:
            test_bar.set_description(f'Evaluating batch {i+1}')
            x_test, y_test = next(test_dataiter)
            x_test, y_test = x_test.to(device), y_test.to(device)

            # compute the model output without calculating the gradient
            with th.no_grad():
                y_pred = model(x_test)

            # calculate loss
            Loss += criterion(y_pred, y_test).item()

            # set information for the bar
            test_bar.set_postfix(evaluate_loss=Loss / (i+1))

            # delete data to release memory
            del x_test, y_test
            th.cuda.empty_cache()

        return Loss / (i+1)
    
# train the nn model
def train(model, train_dataloader, criterion, valid_dataloader=None, MAX_EPOCH=MAX_EPOCH):

    # set the optimizer
    # optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f'{MAX_EPOCH} epochs to train: ')

    for epoch in range(1, MAX_EPOCH+1):

        # enumerate mini batches
        print('epoch {}/{}:'.format(epoch, MAX_EPOCH))
        train_data_size = len(train_dataloader)
        train_dataiter = iter(train_dataloader)
        model.train()

        Total_loss = 0

        # set the bar to check the progress
        with trange(train_data_size) as train_bar:
            for i in train_bar:
                train_bar.set_description(f'Training batch {i+1}')
                x_train, y_train = next(train_dataiter)
                x_train, y_train = x_train.to(device), y_train.to(device)

                # clear the gradients
                optimizer.zero_grad()

                # compute the model output
                y_pred = model(x_train)

                # calculate loss
                train_loss = criterion(y_pred, y_train)

                if th.isnan(train_loss):
                    continue

                Total_loss += train_loss.item()

                # credit assignment
                train_loss.backward()

                # update model weights
                optimizer.step()

                # set information for the bar
                train_bar.set_postfix(train_loss=Total_loss / (i+1))

                # delete data to release memory
                del x_train, y_train
                th.cuda.empty_cache()

            # see whether the trained model after this epoch is the currently best
            if valid_dataloader is not None:
                model.eval()
                loss = evaluate(model, valid_dataloader, criterion)
                model.train()

    return model, Total_loss

# evaluate the performance of the nn model
def evaluate_rank(model, dataloader, criterion, n_stocks):

    # enumerate mini batches
    print('Evaluating ...')
    test_data_size = len(dataloader)
    test_dataiter = iter(dataloader)
    model.eval()

    # initailize the loss
    Loss = 0

    # set the bar to check the progress
    with trange(test_data_size) as test_bar:
        for i in test_bar:
            test_bar.set_description(f'Evaluating batch {i+1}')
            x_test, y_test = next(test_dataiter)
            y_test = th.tensor(n_stocks, requires_grad = False)
            x_test, y_test = x_test.to(device), y_test.to(device)

            # compute the model output without calculating the gradient
            with th.no_grad():
                y_pred = model(x_test)

            # calculate loss
            Loss += criterion(y_pred, y_test).item()

            # set information for the bar
            test_bar.set_postfix(evaluate_loss=Loss / (i+1))

            # delete data to release memory
            del x_test, y_test
            th.cuda.empty_cache()

        return Loss / (i+1)
    
# train the nn model
def train_rank(model, train_dataloader, criterion, n_stocks, valid_dataloader=None, MAX_EPOCH=MAX_EPOCH):

    # set the optimizer
    # optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f'{MAX_EPOCH} epochs to train: ')

    for epoch in range(1, MAX_EPOCH+1):

        # enumerate mini batches
        print('epoch {}/{}:'.format(epoch, MAX_EPOCH))
        train_data_size = len(train_dataloader)
        train_dataiter = iter(train_dataloader)
        model.train()

        Total_loss = 0
        min_loss = math.inf
        best_model = None

        # set the bar to check the progress
        with trange(train_data_size) as train_bar:
            for i in train_bar:
                train_bar.set_description(f'Training batch {i+1}')
                x_train, y_train = next(train_dataiter)
                x_train, y_train = x_train.to(device), y_train.to(device)
                y_train = th.tensor(n_stocks, requires_grad = False)

                # clear the gradients
                optimizer.zero_grad()

                # compute the model output
                y_pred = model(x_train)

                # calculate loss
                train_loss = criterion(y_pred, y_train)

                if th.isnan(train_loss):
                    continue

                Total_loss += train_loss.item()

                # credit assignment
                train_loss.backward()

                # update model weights
                optimizer.step()

                # set information for the bar
                train_bar.set_postfix(train_loss=Total_loss / (i+1))

                # delete data to release memory
                del x_train, y_train
                th.cuda.empty_cache()

            # see whether the trained model after this epoch is the currently best
            if valid_dataloader is not None:
                model.eval()
                loss = evaluate_rank(model, valid_dataloader, criterion, n_stocks)
                if loss <= min_loss:
                    best_model = model
                    min_loss = loss
                model.train()
            else:
                best_model = model

    return best_model, Total_loss

# get the factor by applying nn models to prediction
def get_factor(model, dataloader):
    # enumerate mini batches
    print('Evaluating ...')
    test_data_size = len(dataloader)
    test_dataiter = iter(dataloader)
    model.eval()

    factors = []

    # set the bar to check the progress
    with trange(test_data_size) as test_bar:
        for i in test_bar:
            test_bar.set_description(f'Evaluating batch {i+1}')
            x_test, y_test = next(test_dataiter)
            x_test, y_test = x_test.to(device), y_test.to(device)

            # compute the model output
            with th.no_grad():
                y_pred = model(x_test)
                factors.append(y_pred)

            # delete data to release memory
            del x_test, y_test
            th.cuda.empty_cache()

    # concatenate the data to get the output factors
    factors = th.cat(factors, dim=0).cpu().detach().numpy()

    return factors

# pearson correlation
def IC(x, y):
    vx = x - x.mean()
    vy = y - y.mean()
    cost = (vx * vy).sum() / (math.sqrt((vx ** 2).sum()) * math.sqrt((vy ** 2).sum()))
    return cost

# Weighted pearson correlation
def WIC(logits, target):
    rank = (logits * 1).argsort().argsort()
    weights = 0.5 ** (rank / (len(rank)-1))
    mean_w_x = (weights*logits).sum()
    mean_w_r = (weights*target).sum()
    numera = (weights*logits*target).sum() - mean_w_x * mean_w_r
    var_w_x = (weights*logits**2).sum() - mean_w_x ** 2
    var_w_r = (weights*target**2).sum() - mean_w_r ** 2
    denomi = math.sqrt(abs(var_w_x * var_w_r))
    return numera / denomi

# backtesting
def get_backtest_prep(save_dir: str, factor_dir: str, modelLst: list, loss: str, 
                      start_date: int, end_date: int, bench: str, T: int, weights: str, gap: bool=True):

    # create folder to store the return data
    if not os.path.exists(f'{save_dir}/Metrics/{bench}/{T}'):
        os.makedirs(f'{save_dir}/Metrics/{bench}/{T}')
        
    if gap:
        dirLst = os.listdir(f'{save_dir}/Data/Return/allAshare/T+1_T+{T+1}')
    else:
        dirLst = os.listdir(f'{save_dir}/Data/Return/allAshare/T_T+{T}')

    # get the directory of stock return data
    dirLst = [int(dir[:8]) for dir in dirLst]
    dirLst = np.array(dirLst)
    dirLst = dirLst[(dirLst >= start_date) & (dirLst <= end_date)]

    # load all the predicted factor based on different datasets
    factorDFLst = []
    for model in modelLst:
        for unit in ['Stock', 'Factor1', 'Factor2']:
            factor_df = pd.DataFrame()
            with trange(len(dirLst)) as dir_bar:    
                for i in dir_bar:
                    dir_i = dirLst[i]
                    dir_bar.set_description(f'Loading factor data on unit {unit} predicted by {model} on trading date {dir_i}')
                    score_i = pd.read_csv(f'{factor_dir}/{loss}/{bench}/{unit}/{model}/{dir_i}.csv')
                    factor_df = factor_df.append(score_i)

            return_df = factor_df.loc[:,['id', 'date', 'return']].copy()
            del factor_df['return']
            factor_df.set_index(keys=['id', 'date'], inplace=True)
            factor_df[f'{model}_{unit}_score'] = factor_df.mean(axis=1)
            factor_df = factor_df[f'{model}_{unit}_score']
            factorDFLst.append(factor_df)
    
    # merge all the factor data
    df = reduce(lambda df1, df2: pd.merge(df1, df2, on=['id', 'date'], how='inner'), factorDFLst)
    df.reset_index(drop=False, inplace=True)

    # first calculate IC to determine the direction of factors, then concatenate factors by equal weights
    if weights == 'equal':

        def get_IC(x):
            return IC(x.iloc[:,0], x.iloc[:,1])

        df = pd.merge(df, return_df, on=['id', 'date'])
        feature_cols = [col for col in df.columns if not col in ['id', 'date', 'return', 'close']]
        ICLst = []
        for f in feature_cols:
            print(f'Calculate IC of factor {f}')
            temp = df.loc[:,['date', f, 'return']]
            temp = temp.groupby('date')[[f,'return']].apply(get_IC).reset_index()
            temp.rename(columns={0: f'{f}_IC'}, inplace=True)
            ICLst.append(temp)
        
        IC_df = reduce(lambda df1, df2: pd.merge(df1, df2, on='date'), ICLst)
        IC_df = IC_df.mean(axis=0)
        IC_df = np.sign(IC_df)
        for f in feature_cols:
            df[f] *= IC_df[f'{f}_IC']
        df['score'] = df[feature_cols].mean(axis=1)
    # first calculate IC to determine the direction of factors, then concatenate factors by rolling ICIR
    elif weights == 'ICIR':

        def get_IC(x):
            return IC(x.iloc[:,0], x.iloc[:,1])

        df = pd.merge(df, return_df, on=['id', 'date'])
        print(df)
        feature_cols = [col for col in df.columns if not col in ['id', 'date', 'return', 'close']]
        ICLst = []
        for f in feature_cols:
            print(f'Calculate IC of factor {f}')
            temp = df.loc[:,['date', f, 'return']]
            temp = temp.groupby('date')[[f,'return']].apply(get_IC).reset_index()
            temp.rename(columns={0: f'{f}_IC'}, inplace=True)
            ICLst.append(temp)
        
        IC_df = reduce(lambda df1, df2: pd.merge(df1, df2, on='date'), ICLst)
        featureIC_cols = [f'{col}_IC' for col in feature_cols]
        IC_df[featureIC_cols] = IC_df[featureIC_cols].rolling(252).mean() / IC_df[featureIC_cols].rolling(252).std()
        IC_df.dropna(subset=featureIC_cols, inplace=True)
        df = pd.merge(df, IC_df, on='date', how='inner')
        for f in feature_cols:
            df[f] *= df[f'{f}_IC']
        df['sum'] = df[featureIC_cols].abs().sum(axis=1)
        df[feature_cols] = df[feature_cols].div(df["sum"], axis=0)
        df['score'] = df[feature_cols].sum(axis=1)
    # first calculate weighted IC to determine the direction of factors, then concatenate factors by rolling weighted ICIR
    elif weights == 'weighted_ICIR':

        def get_WIC(x):
            return WIC(x.iloc[:,0], x.iloc[:,1])

        df = pd.merge(df, return_df, on=['id', 'date'])
        feature_cols = [col for col in df.columns if not col in ['id', 'date', 'return', 'close']]
        ICLst = []
        for i in range(len(feature_cols)):
            f = feature_cols[i]
            print(f'Calculate WIC of factor {f}')
            temp = df.loc[:,['date', f, 'return']]
            temp = temp.groupby('date')[[f,'return']].apply(get_WIC).reset_index()
            temp.rename(columns={0: f'{f}_WIC'}, inplace=True)
            ICLst.append(temp)
            
        IC_df = reduce(lambda df1, df2: pd.merge(df1, df2, on='date'), ICLst)
        featureIC_cols = [f'{col}_WIC' for col in feature_cols]
        IC_df[featureIC_cols] = IC_df[featureIC_cols].rolling(252).mean() / IC_df[featureIC_cols].rolling(252).std()
        IC_df.dropna(subset=featureIC_cols, inplace=True)
        df = pd.merge(df, IC_df, on='date', how='inner')
        for f in feature_cols:
            df[f] *= df[f'{f}_WIC'].abs()
        df['sum'] = df[featureIC_cols].abs().sum(axis=1)
        df[feature_cols] = df[feature_cols].div(df["sum"], axis=0)
        df['score'] = df[feature_cols].sum(axis=1)

    model_name = '_'.join(modelLst)

    df = df.loc[:,['id', 'date', 'score', 'return']]
    df.to_csv(f'{save_dir}/Metrics/{bench}/{T}/{model_name}_{loss}_{weights}_{start_date}_{end_date}.csv')


# backtesting
def backtest(save_dir: str, modelLst: list, loss: str, start_date: int, end_date: int, bench: str, T: int, weights: str,
             method: str, thres, sign: int=1, gap: bool=True):

    # create folder to store the return data
    if not os.path.exists(f'{save_dir}/Report/{bench}/{T}'):
        os.makedirs(f'{save_dir}/Report/{bench}/{T}')

    name = '_'.join(modelLst)

    # read scores and true return data
    df = pd.read_csv(f'{save_dir}/Metrics/{bench}/{T}/{name}_{loss}_{weights}_{start_date}_{end_date}.csv')
    dates = np.sort(pd.unique(df['date']))
    
    if gap:
        dates = [dates[i] for i in range(0,len(dates),T+1)]
    else:
        dates = [dates[i] for i in range(0,len(dates),T)]

    df = df[df['date'].isin(dates)]
    return_df = df.loc[:,['id', 'date', 'return']].copy()
    df = df.loc[:,['date', 'score', 'return']].copy()

    # get the return of benchmark
    bench_df = return_df.groupby(by='date')['return'].mean().reset_index()
    bench_df.rename(columns={'return': 'benchmark'}, inplace=True)

    # calculate return of top N stocks based on their predicted score
    def topN(x):
        score = x['score'].values
        ret = x['return'].values
        if sign == 1:
            ids = np.argsort(score)[::-1]
        elif sign == -1:
            ids = np.argsort(score)
        else:
            raise ValueError(f'The parameter sign should be -1/1, get {sign} instead')
        if method == 'topN':
            return ret[ids[:thres]].mean()
        elif method == 'Percent':
            return ret[ids[:int(ret.shape[0]*thres)]].mean()
        else:
            raise ValueError(f'The parameter should be topN/Percent, get {method} instead')

    df = df.groupby('date').apply(topN)
    df.name = 'return'
    df = df.to_frame()
    df.reset_index(inplace=True)
    portfolio = pd.merge(df, bench_df, on='date')
    portfolio['date'] = portfolio['date'].apply(lambda date_i: pd.Timestamp(int(str(date_i)[:4]), int(str(date_i)[4:6]), int(str(date_i)[6:])))
    portfolio.set_index('date', inplace=True)
    print(portfolio)

    # create the report under the path
    if gap:
        report_dir = f'{save_dir}/Report/{bench}/{T}/{name}_{loss}_{start_date}_{end_date}_{weights}_T+1_T+{T+1}.html'
    else:
        report_dir = f'{save_dir}/Report/{bench}/{T}/{name}_{loss}_{start_date}_{end_date}_{weights}_T_T+{T}.html'

    qs.reports.html(portfolio['return'], portfolio['benchmark'],
        title=f'Report of long-short portfolio with factor predicted by {name}',
        output=report_dir)
    
    print('Report saved in %s' % (report_dir))


def get_metrics(save_dir: str, factor_dir: str, modelLst: list, start_date: int, end_date: int, bench: str, T: int, name=None):

    # create folder to store the return data
    if not os.path.exists(f'{save_dir}/Metrics/{bench}/{T}'):
        os.makedirs(f'{save_dir}/Metrics/{bench}/{T}')

    # dirLst = os.listdir(f'{save_dir}/Data/Return/allAshare/{T}/{dir_i}.csv')
    # dirLst = [int(dir[:8]) for dir in dirLst]
    # dirLst = np.array(dirLst)
    # dirLst = dirLst[(dirLst >= start_date) & (dirLst <= end_date)]

    # return_df = pd.DataFrame()
    # with trange(len(dirLst)) as dir_bar:    
    #     for i in dir_bar:
    #         dir_i = dirLst[i]
    #         dir_bar.set_description(f'Loading return data of T+{T} on trading date {dir_i}')
    #         return_i = pd.read_csv(f'{save_dir}/Data/Return/allAshare/{dir_i}.csv')
    #         return_df = return_df.append(return_i)

    # return_df.reset_index(drop=True, inplace=True)
    # return_df = return_df[['date', 'return']].groupby(by='date')['return'].mean().reset_index()
    # return_df.rename(columns={'return': 'benchmark'}, inplace=True)

    for model in modelLst:

        for unit in ['Stock', 'Factor1', 'Factor2']:
            if not os.path.exists(f'{save_dir}/Metrics/{bench}/{T}/{model}_{unit}_IC_{start_date}_{end_date}.csv'):
                score_df = pd.DataFrame()
                dirLst = os.listdir(f'{factor_dir}/{bench}/{unit}/{model}')
                with trange(len(dirLst)) as dir_bar:    
                    for i in dir_bar:
                        dir_i = dirLst[i]
                        dir_bar.set_description(f'Loading return data of T+{T} on trading date {dir_i}')
                        score_i = pd.read_csv(f'{factor_dir}/{bench}/{unit}/{model}/{dir_i}')
                        score_df = score_df.append(score_i)
                
                f_cols = [col for col in score_df.columns if not col in ['id', 'date', 'return']]
                score_df['score'] = score_df[f_cols].mean(axis=1)
                score_df = score_df.loc[:,['date', 'return', 'score']]

                # calculate the IC (pearson correlation)
                def IC(x):
                    return sp.pearsonr(x['score'], x['return'])[0]
                # calculate the Rank IC (spearman correlation)
                def RankIC(x):
                    return sp.spearmanr(x['score'], x['return'])[0]
                
                ICdf = score_df.groupby('date').apply(IC)
                RankICdf = score_df.groupby('date').apply(RankIC)
                ICdf.name, RankICdf.name = 'IC', 'Rank IC'
                ICdf, RankICdf = ICdf.to_frame().reset_index(), RankICdf.to_frame().reset_index()

                if not name is None:
                    ICdf.to_csv(f'{save_dir}/Metrics/{bench}/{T}/{model}_{unit}_IC_{start_date}_{end_date}.csv', index=False)
                    RankICdf.to_csv(f'{save_dir}/Metrics/{bench}/{T}/{model}_{unit}_RankIC_{start_date}_{end_date}.csv', index=False)
                else:
                    ICdf.to_csv(f'{save_dir}/Metrics/{bench}/{T}/{model}_{unit}_{name}_IC_{start_date}_{end_date}.csv', index=False)
                    RankICdf.to_csv(f'{save_dir}/Metrics/{bench}/{T}/{model}_{unit}_{name}_RankIC_{start_date}_{end_date}.csv', index=False)
            else:
                if not name is None:
                    ICdf = pd.read_csv(f'{save_dir}/Metrics/{bench}/{T}/{model}_{unit}_IC_{start_date}_{end_date}.csv', index_col=None)
                    RankICdf = pd.read_csv(f'{save_dir}/Metrics/{bench}/{T}/{model}_{unit}_RankIC_{start_date}_{end_date}.csv', index_col=None)
                else:
                    ICdf = pd.read_csv(f'{save_dir}/Metrics/{bench}/{T}/{model}_{unit}_{name}_IC_{start_date}_{end_date}.csv', index_col=None)
                    RankICdf = pd.read_csv(f'{save_dir}/Metrics/{bench}/{T}/{model}_{unit}_{name}_RankIC_{start_date}_{end_date}.csv', index_col=None)

            ic = ICdf['IC'].mean()
            ir = ICdf['IC'].mean() / ICdf['IC'].std()
            rankic = RankICdf['Rank IC'].mean()

            print(f'The mean IC of factor predicted by {model} based on dataset {unit} is: {ic}')
            print(f'The IR of factor predicted by {model} based on dataset {unit} is: {ir}')
            print(f'The mean Rank IC of factor predicted by based on dataset {unit} {model} is: {rankic}')


def barra_neutral(factor_dir: str, barra_dir: str, trDays_dir: str, featureLst: list, modelLst: list, start_date: int, 
                  end_date: int, bench: str, loss: str, f_method: str='large'):

    # load the trading date data within the corresponding time interval
    timeLst = read_pkl5_data(trDays_dir)
    timeLst = timeLst.index.to_numpy()
    timeLst = timeLst[(start_date <= timeLst) & (timeLst <= end_date)]

    # load all the predicted factor based on different datasets
    factorDFLst = []
    for model in modelLst:
        for unit in ['Stock', 'Factor1', 'Factor2']:
            factor_df = pd.DataFrame()
            with trange(len(timeLst)) as time_bar:    
                for i in time_bar:
                    time_i = timeLst[i]
                    time_bar.set_description(f'Loading factor data on unit {unit} predicted by {model} on trading date {time_i}')
                    score_i = pd.read_csv(f'{factor_dir}/{loss}/{bench}/{unit}/{model}/{time_i}.csv')
                    factor_df = factor_df.append(score_i)

            return_df = factor_df.loc[:,['id', 'date', 'return']].copy()
            del factor_df['return']
            factor_df.set_index(keys=['id', 'date'], inplace=True)
            if f_method == 'large':
                factor_df[f'{model}_{unit}_score'] = factor_df.mean(axis=1)
                factor_df = factor_df[f'{model}_{unit}_score']
                factorDFLst.append(factor_df)
            elif f_method == 'small':
                for f_col in factor_df.columns:
                    temp = factor_df[f_col]
                    factorDFLst.append(temp)
            else:
                raise ValueError(f'The parameter f_method should be large/small, get {f_method} instead')
    
    # merge all the factor data
    df = reduce(lambda df1, df2: pd.merge(df1, df2, on=['id', 'date'], how='inner'), factorDFLst)
    f_cols = list(df.columns)
    df.reset_index(drop=False, inplace=True)

    BarraLst = [df]
    featureNameLst = [f.split('_')[0] for f in featureLst]
    for i in range(len(featureLst)):
        f = featureLst[i]
        f_name = featureNameLst[i]
        barra_i = stack_pkl_data(f'{barra_dir}/{f}', start_date, f_name)
        barra_i = barra_i.loc[barra_i['date'] <= end_date]
        BarraLst.append(barra_i)

    df = reduce(lambda df1, df2: pd.merge(df1, df2, on=['id', 'date'], how='left'), BarraLst)
    df[featureNameLst] = df[featureNameLst].fillna(0)

    for f in f_cols:
        y = df[f].values.astype(float)
        print(df[featureNameLst].values)
        X = df[featureNameLst].values.astype(float)
        model = sm.OLS(y, X)
        results = model.fit()
        df[f] = results.resid
        print(results.resid)

    df = df.loc[:,['id', 'date']+f_cols]
    df = pd.merge(df, return_df, on=['id', 'date'])

    # store the factor data
    tdateLst = np.sort(pd.unique(df['date']))
    with trange(len(tdateLst)) as date_bar:
        for i in date_bar:
            date_i = tdateLst[i]
            date_bar.set_description(f'Saving data on date {date_i}')

            df_i = df.loc[df['date'] == date_i]
            for f in f_cols:
                fLst = f.split('_')
                model, unit = fLst[0], fLst[1]

                if not os.path.exists(f'{factor_dir}/{loss}/Neutral/{bench}/{unit}/{model}/{f_method}'):
                    os.makedirs(f'{factor_dir}/{loss}/Neutral/{bench}/{unit}/{model}/{f_method}')

                factor_i = df_i.loc[:,['id', 'date', f]]
                factor_i.to_csv(f'{factor_dir}/{loss}/Neutral/{bench}/{unit}/{model}/{f_method}/{date_i}.csv', index=False)


# The linear model to maximum the IC (pearson correlation) between predicted and true stock return
class MaxICLinear(nn.Module):
    def __init__(self, X_size, f_size) -> None:
        self.X_size = X_size
        self.f_size = f_size

        norm_size = int(self.X_size / self.f_size)
        self.fc_lst = [nn.Linear(f_size, 1) for _ in range(norm_size)]
        self.linear = nn.Linear(norm_size, 1)

        self.norm_1 = nn.BatchNorm1d(norm_size)
        self.norm_2 = nn.BatchNorm1d(1)

    def forward(self, x):
        X_lst = []
        idxLst = list(range(0, self.X_size, self.f_size))

        for i in range(len(idxLst)):
            idx = idxLst[i]
            X_i = x[:,idx:idx+self.f_size]
            X_i = self.fc_lst[i](x)
            X_lst.append(X_i)

        out = th.cat(X_lst, dim=1)
        out = self.norm_1(out)
        out = self.linear(out)
        out = self.norm_2(out)
        return out.reshape(-1)


# combine the result of DL model
def combine_neutral_factor(save_dir: str, factor_dir: str, trDays_dir: str, start_year: int, end_year: int, train_window: int, modelLst: list,
    bench: str, loss: str, f_method: str, combine: str, gap: bool):

    # create folder to store the return data
    if not os.path.exists(f'{save_dir}/Model/Neutral/{bench}'):
        os.makedirs(f'{save_dir}/Model/Neutral/{bench}')

    trDate = read_pkl5_data(trDays_dir)
    trDate = trDate.index.to_numpy()
    trDate = trDate[(int(f'{start_year-1}0101') <= trDate) & (trDate <= int(f'{end_year}1231'))]

    # load the stock return data
    return_df = pd.DataFrame()

    with trange(len(trDate)) as date_bar:
        for i in date_bar:
            date_i = trDate[i]
            date_bar.set_description(f'Loading data on date {date_i}')

            if gap:
                try:
                    return_i = pd.read_csv(f'{save_dir}/Data/Return/allAshare/T+1_T+{T+1}/{date_i}.csv')
                except:
                    print(f'No such file: {save_dir}/Data/Return/allAshare/T+1_T+{T+1}/{date_i}.csv')
                    continue
            else:
                try:
                    return_i = pd.read_csv(f'{save_dir}/Data/Return/allAshare/T_T+{T}/{date_i}.csv')
                except:
                    print(f'No such file: {save_dir}/Data/Return/allAshare/T_T+{T}/{date_i}.csv')

            return_df = return_df.append(return_i)

    model_name = '_'.join(modelLst)

    if not os.path.exists(f'{save_dir}/Factor_Neutral/{loss}/{bench}/{model_name}/{combine}'):
        os.makedirs(f'{save_dir}/Factor_Neutral/{loss}/{bench}/{model_name}/{combine}')

    factor = pd.DataFrame()

    # iterately train and predict the factor
    for i in range(start_year, end_year-train_window+1):
        
        # Loading training data within the training window
        train_data_stock, train_data_factor1, train_data_factor2 = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        train_dates = trDate[(trDate >= int(f'{i}0101') & (trDate < int(f'{i+train_window}0101')))]
        
        print(f'Loading training data from year {i} year {i+train_window-1}')
        # load the training dataset
        with trange(len(train_dates)) as train_bar:
            for i in train_bar:
                date_i = train_dates[i]
                train_bar.set_description(f'Loading training data on date {date_i}')
                for model in modelLst:

                    train_stock_i = pd.read_csv(f'{factor_dir}/{loss}/{bench}/Stock/{model}/{f_method}/{date_i}.csv')
                    train_factor1_i = pd.read_csv(f'{factor_dir}/{loss}/{bench}/Factor1/{model}/{f_method}/{date_i}.csv')
                    train_factor2_i = pd.read_csv(f'{factor_dir}/{loss}/{bench}/Factor2/{model}/{f_method}/{date_i}.csv')

                    train_data_stock = train_data_stock.append(train_stock_i)
                    train_data_factor1 = train_data_factor1.append(train_factor1_i)
                    train_data_factor2 = train_data_factor2.append(train_factor2_i)

        train_data_stock.set_index(['id', 'date'], inplace=True)
        train_data_factor1.set_index(['id', 'date'], inplace=True)
        train_data_factor2.set_index(['id', 'date'], inplace=True)

        train_data = [train_data_stock, train_data_factor1, train_data_factor2]
        train_data = pd.concat(train_data, axis=1)
        train_X = train_data.values
        train_data = pd.merge(train_data, return_df, on=['id', 'date'])
        train_Y = train_data['return'].values

        test_data_stock, test_data_factor1, test_data_factor2 = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        test_dates = trDate[(trDate >= int(f'{i+train_window}0101')) & (trDate <= int(f'{i+train_window}1231'))]

        # load the test dataset
        with trange(len(test_dates)) as test_bar:
            for i in test_bar:
                date_i = test_dates[i]
                test_bar.set_description(f'Loading test data on date {date_i}')
                for model in modelLst:

                    test_stock_i = pd.read_csv(f'{factor_dir}/{loss}/{bench}/Stock/{model}/{f_method}/{date_i}.csv')
                    test_factor1_i = pd.read_csv(f'{factor_dir}/{loss}/{bench}/Factor1/{model}/{f_method}/{date_i}.csv')
                    test_factor2_i = pd.read_csv(f'{factor_dir}/{loss}/{bench}/Factor2/{model}/{f_method}/{date_i}.csv')

                    test_data_stock = test_data_stock.append(test_stock_i)
                    test_data_factor1 = test_data_factor1.append(test_factor1_i)
                    test_data_factor2 = test_data_factor2.append(test_factor2_i)

        test_info = test_data[['id', 'date']]
        test_data_stock.set_index(['id', 'date'], inplace=True)
        test_data_factor1.set_index(['id', 'date'], inplace=True)
        test_data_factor2.set_index(['id', 'date'], inplace=True)

        test_data = [test_data_stock, test_data_factor1, test_data_factor2]
        test_data = pd.concat(test_data, axis=1)
        test_X = test_data.values
        test_data = pd.merge(test_data, return_df, on=['id', 'date'])
        test_Y = test_data['return'].values

        # 'Factor': Use linear or lightGBM model to train and predict factor
        # 'ZScore': Combine factors in terms of zscore
        
        # model = combine_factor(train_data, modelLst, 'train', combine, 60)
        # pred_Y = combine_factor(test_data, modelLst, 'test', combine, 60, model)

        # test_info['factor'] = pred_Y
        # factor = factor.append(test_info)
        if combine == 'maxIC':
            train_X, train_Y = th.from_numpy(train_X), th.from_numpy(train_Y)
            test_X, test_Y = th.from_numpy(test_X), th.from_numpy(test_Y)
            train_dataset = Data(train_X, train_Y)
            test_dataset = Data(test_X, test_Y)
            train_dataloader = DataLoader(train_dataset, FACTOR_BATCH_SIZE, shuffle=False)
            test_dataloader = DataLoader(test_dataset, FACTOR_BATCH_SIZE, shuffle=False)
            if os.path.exists(f'{save_dir}/Model/Neutral/{bench}/{model_name}_{loss}_{i}_{i+train_window-1}.m'):
                Best_model = joblib.load(f'{save_dir}/Model/Neutral/{bench}/{model_name}_{loss}_{i}_{i+train_window-1}.m')
                Best_model.to(device=device)
                Best_model.eval()
            else:
                model = MaxICLinear(train_X.size(1), 60)
                Best_model, _ = train(model, train_dataloader, IC_loss, None, MAX_EPOCH)
                joblib.dump(Best_model, f'{save_dir}/Model/Neutral/{bench}/{model_name}_{loss}_{i}_{i+train_window-1}.m')

            # get the corresponding factor data
            pred_Y = get_factor(Best_model, test_dataloader)
            test_info['return'] = test_Y
            test_info[f'{model_name}_score'] = pred_Y

        if combine == 'Zscore':
            test_data.loc[:,:] = sp.stats.zscore(test_data.values, axis=0)
            pred_Y = test_data.mean(axis=1)
            factor = factor.append(pred_Y)

    factor.reset_index(drop=True, inplace=True)

    # save the final result
    dates = np.sort(pd.unique(factor['date']))
    with trange(len(dates)) as date_bar:
        for i in date_bar:
            date_i = dates[i]
            date_bar.set_description(f'Saving factor data on date {date_i}')
            factor_i = factor.loc[factor['date'] == date_i]
            factor_i.to_csv(f'{save_dir}/Factor_Neutral/{loss}/{bench}/{model_name}/{combine}/{date_i}.csv', index=False)


# # this loss is the sharpe ratio of the long position portfolio
# def SP_loss(logits, target, thres=100, sign=-1):
#     """
#     logits: (N, 1)
#     target: (N, T)
#     """
#     if sign == 1:
#         idx = th.argsort(logits)[-thres:]
#     elif sign == -1:
#         idx = th.argsort(logits)[:thres]
#     else:
#         raise ValueError(f'The parameter sign should be -1/1, get {idx} instead.')
    
#     target = target[idx, :]
#     target = target.mean()
#     return sharpe_ratio(target)

# # this loss is basically the SP loss with penalty on correlation between different factors
# def SP_loss_penalty(logits, target, thres=100, sign=-1, phi=1e-2):
#     c = logits.mean(dim=1)
#     penalty = 0
#     for i in range(logits.size(1)):
#         for j in range(i+1, logits.size(1)):
#             penalty += pearson_corr(logits[:,i], logits[:,j])

#     loss = SP_loss(c, target, thres, sign)
#     return loss + phi*penalty

# # define a function to calculate the Sharpe Ratio
# def sharpe_ratio(returns, risk_free_rate: float=0):
#     """
#     Calculate the Sharpe Ratio of a portfolio.

#     Parameters:
#     returns: The portfolio returns.
#     risk_free_rate (float): The annual risk-free rate.

#     Returns:
#     float: The Sharpe Ratio of the portfolio.
#     """
#     excess_returns = returns - risk_free_rate
#     return np.sqrt(returns.shape[0]) * (excess_returns.mean() / excess_returns.std())


# # define a function to calculate the win ratio
# def win_ratio(returns):
#     """
#     Calculate the win ratio of a trading strategy.

#     Parameters:
#     returns (list): A list of returns for each trade.

#     Returns:
#     float: The win ratio of the trading strategy.
#     """
#     wins = [r for r in returns if r > 0]
#     return len(wins) / len(returns)


# # The linear model to maximum the IC (pearson correlation) between predicted and true stock return
# class MaxICLinear(nn.Module):
#     def __init__(self, X_size, f_size) -> None:
#         self.X_size = X_size
#         self.f_size = f_size

#         norm_size = int(self.X_size / self.f_size)
#         self.fc_lst = [nn.Linear(f_size, 1) for _ in range(norm_size)]
#         self.linear = nn.Linear(norm_size, 1)

#         self.norm_1 = nn.BatchNorm1d(norm_size)
#         self.norm_2 = nn.BatchNorm1d(1)

#     def forward(self, x):
#         X_lst = []
#         idxLst = list(range(0, self.X_size, self.f_size))

#         for i in range(len(idxLst)):
#             idx = idxLst[i]
#             X_i = x[:,idx:idx+self.f_size]
#             X_i = self.fc_lst[i](x)
#             X_lst.append(X_i)

#         out = th.cat(X_lst, dim=1)
#         out = self.norm_1(out)
#         out = self.linear(out)
#         out = self.norm_2(out)
#         return out.reshape(-1)

# # combine the factor to get the predicted return
# def combine_factor(df: pd.DataFrame, mode: str, combine: str, f_size: int, modelFitted=None):
#     # choose the method to combine factors
#     if combine == 'Zscore':
#         # choose the mode of this functiom. train: train a model to combine factor; test: use a pre-trained model to combine factor
#         if mode == 'train':
#             train_Y = df['return'].values
#             del df['return']
#             train_X = df.values
#             model = LGBMRegressor()
#             model = model.fit(train_X, train_Y)
#             return model
#         elif mode == 'test':
#             del df['return']
#             test_X = df.values
#             pred_Y = modelFitted.predict(test_X)
#             return pred_Y
#         else:
#             raise ValueError('The parameter mode should be train / test, get %s instead' % mode)
#     # combining factors in terms of their z-scores
#     elif combine == 'MaxIC':
#         # choose the mode of this functiom. train: train a model to combine factor; test: use a pre-trained model to combine factor
#         if mode == 'train':
#             train_Y = df['return'].values
#             del df['return']
#             train_X = df.values
#             train_X, train_Y = th.from_numpy(train_X), th.from_numpy(train_Y)
#             model = MaxICLinear(train_X.view(1), f_size)
#             train_data = Data(train_X, train_Y)
#             train_dataloader = DataLoader(train_data)
#             model = train(model, train_dataloader, IC_loss)
#             return model
#         elif mode == 'test':
#             del df['return']
#             test_X = df.values
#             test_X = th.from_numpy(test_X)
#             pred_Y = modelFitted(test_X)
#             return pred_Y
#         else:
#             raise ValueError('The parameter mode should be train / test, get %s instead' % mode)
#     else:
#         raise ValueError('The parameter combine should be lgbols / ZScore, get %s instead' % combine)

# # combine the result of DL model
# def combine_factor_train_and_test(save_dir: str, trDays_dir: str, start_year: int, end_year: int, train_window: int, modelLst: list,
#     combine: str, bench: str):

#     trDate = read_pkl5_data(trDays_dir)
#     trDate = trDate.index.to_numpy()

#     model_name = '_'.join(modelLst)

#     if not os.path.exists(f'{save_dir}/Factor/{bench}/Combine/{model_name}/{combine}'):
#         os.makedirs(f'{save_dir}/Factor/{bench}/Combine/{model_name}/{combine}')

#     factor = pd.DataFrame()

#     # iterately train and predict the factor
#     for i in range(start_year, end_year-train_window+1):
        
#         # Loading training data within the training window
#         train_data_stock, train_data_factor1, train_data_factor2 = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
#         train_dates = trDate[(trDate >= int(f'{i}0101') & (trDate < int(f'{i+train_window}0101')))]
        
#         if combine == 'maxIC':
#             print(f'Loading training data from year {i} year {i+train_window-1}')
#             # load the training dataset
#             with trange(len(train_dates)) as train_bar:
#                 for i in train_bar:
#                     date_i = train_dates[i]
#                     train_bar.set_description(f'Loading training data on date {date_i}')
#                     for model in modelLst:

#                         train_stock_i = pd.read_csv(f'{save_dir}/Factor/{bench}/Stock/{model}/{date_i}.csv')
#                         train_factor1_i = pd.read_csv(f'{save_dir}/Factor/{bench}/Factor1/{model}/{date_i}.csv')
#                         train_factor2_i = pd.read_csv(f'{save_dir}/Factor/{bench}/Factor2/{model}/{date_i}.csv')

#                         train_data_stock = train_data_stock.append(train_stock_i)
#                         train_data_factor1 = train_data_factor1.append(train_factor1_i)
#                         train_data_factor2 = train_data_factor2.append(train_factor2_i)

#             train_data_stock.set_index(['id', 'date'], inplace=True)
#             train_data_factor1.set_index(['id', 'date'], inplace=True)
#             train_data_factor2.set_index(['id', 'date'], inplace=True)

#             train_data = [train_data_stock, train_data_factor1, train_data_factor2]
#             train_data = pd.concat(train_data, axis=1)

#         test_data_stock, test_data_factor1, test_data_factor2 = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
#         test_dates = trDate[(trDate >= int(f'{i+train_window}0101')) & (trDate <= int(f'{i+train_window}1231'))]

#         # load the test dataset
#         with trange(len(test_dates)) as test_bar:
#             for i in test_bar:
#                 date_i = test_dates[i]
#                 test_bar.set_description(f'Loading test data on date {date_i}')
#                 for model in modelLst:

#                     test_stock_i = pd.read_csv(f'{save_dir}/Factor/{bench}/Stock/{model}/{date_i}.csv')
#                     test_factor1_i = pd.read_csv(f'{save_dir}/Factor/{bench}/Factor1/{model}/{date_i}.csv')
#                     test_factor2_i = pd.read_csv(f'{save_dir}/Factor/{bench}/Factor2/{model}/{date_i}.csv')

#                     test_data_stock = test_data_stock.append(test_stock_i)
#                     test_data_factor1 = test_data_factor1.append(test_factor1_i)
#                     test_data_factor2 = test_data_factor2.append(test_factor2_i)

#         test_info = test_data[['id', 'date']]
#         test_data_stock.set_index(['id', 'date'], inplace=True)
#         test_data_factor1.set_index(['id', 'date'], inplace=True)
#         test_data_factor2.set_index(['id', 'date'], inplace=True)

#         test_data = [test_data_stock, test_data_factor1, test_data_factor2]
#         test_data = pd.concat(test_data, axis=1)

#         # 'Factor': Use linear or lightGBM model to train and predict factor
#         # 'ZScore': Combine factors in terms of zscore
        
#         # model = combine_factor(train_data, modelLst, 'train', combine, 60)
#         # pred_Y = combine_factor(test_data, modelLst, 'test', combine, 60, model)

#         # test_info['factor'] = pred_Y
#         # factor = factor.append(test_info)
#         if combine == 'Zscore':
#             test_data.loc[:,:] = sp.stats.zscore(test_data.values, axis=0)
#             pred_Y = test_data.mean(axis=1)
#             factor = factor.append(pred_Y)

#     factor.reset_index(drop=True, inplace=True)

#     # save the final result
#     dates = np.sort(pd.unique(factor['date']))
#     with trange(len(dates)) as date_bar:
#         for i in date_bar:
#             date_i = dates[i]
#             date_bar.set_description(f'Saving factor data on date {date_i}')
#             factor_i = factor.loc[factor['date'] == date_i]
#             factor_i.to_csv(f'{save_dir}/Factor/{bench}/Combine/{model_name}/{combine}/{date_i}.csv', index=False)

# # read the high frequency data
# def read_high_freq_data(save_dir: str, high_freq_dir: str, member_dir: str, start_date: int, end_date: int, bench: str):
#     # create folder to store the high frequency data
#     if not os.path.exists(f'{save_dir}/Data/High_Freq/{bench}'):
#         os.makedirs(f'{save_dir}/Data/High_Freq/{bench}')

#     if os.listdir(f'{save_dir}/Data/High_Freq/{bench}'):
#         exist_date = max(os.listdir(f'{save_dir}/Data/High_Freq/{bench}'))
#         exist_date = int(exist_date[:8])
#         start_date = max(exist_date, start_date)

#     dirLst = os.listdir(high_freq_dir)
#     dirLst = [i for i in dirLst if int(i[:-4]) >= start_date and int(i[:-4]) <= end_date]
#     # read the high frequency data
#     df = pd.DataFrame()
#     with trange(len(dirLst)) as dir_bar:
#         for i in dir_bar:
#             dir = dirLst[i]
#             date_i = int(dir[:-4])
#             dir_bar.set_description(f'Loading high frequency data on date {date_i}')
#             df_i = read_pkl_data(f'{high_freq_dir}/{dir}')
#             df_i = df_i.loc[:,['date', 'stock_code', 'time', 'open', 'high', 'low', 'close', 'volume', 'turnover']]
#             df_i.rename(columns={'stock_code': 'id'}, inplace=True)
#             df = df.append(df_i)
#     # get the member from the stock index
#     if bench != 'allAshare':
#         member = read_member(member_dir, start_date, end_date)
#         df = pd.merge(member, df, on=['id', 'date'])
#     # store the stock data by dates
#     dateLst = np.sort(pd.unique(df['date']))
#     with trange(len(dateLst)) as date_bar:    
#         for i in date_bar:
#             date_i = dateLst[i]
#             date_bar.set_description(f'Saving high frequency data on trading date {date_i}')
#             df_i = df.loc[df['date'] == date_i]
#             df_i.to_csv(f'{save_dir}/Data/High_Freq/{bench}/{date_i}.csv', index=False)


# # fill the NA value by the average of its neighbors
# def fill_na(df: pd.DataFrame, cols: list):
#     df[cols] = df[cols].interpolate(method='linear', limit_direction='both')
#     return df

# # clip the extreme values within 3 standard deviation 
# def clip_extreme(df: pd.DataFrame, cols):
#     for col in cols:
#         avg = df[col].mean()
#         standard = df[col].std()
#         df[col].clip(avg - 3*standard, avg + 3*standard, inplace=True)
#     return df

# # normalize all the features
# def normalize(df, cols):
#     for col in cols:
#         avg = df[col].mean()
#         standard = df[col].std()
#         df[col] = (df[col] - avg) / standard
#     return df

# # read the Alpha158/Alpha360 from qlib
# def read_alpha_data(save_dir: str, alpha_dir: str, member_dir: str, start_date: int, end_date: int, bench: str):
#     # create folder to store alpha data
#     if not os.path.exists(f'{save_dir}/Data/Qlib_Alpha/{bench}'):
#         os.makedirs(f'{save_dir}/Data/Qlib_Alpha/{bench}')

#     if os.listdir(f'{save_dir}/Data/Qlib_Alpha/{bench}'):
#         exist_date = max(os.listdir(f'{save_dir}/Data/Qlib_Alpha/{bench}'))
#         exist_date = int(exist_date[:8])
#         start_date = max(exist_date, start_date)

#     print('loading alpha data ...')
#     data = read_pkl_data(alpha_dir)

#     columns, labels, features = data['columns'], data['labels'], data['features']
#     columns.remove('LABEL0')

#     labels.reset_index(drop=False, inplace=True)
#     features.reset_index(drop=False, inplace=True)

#     features.rename(columns={'datetime': 'date', 'instrument': 'id'}, inplace=True)
#     labels.rename(columns={'datetime': 'date', 'instrument': 'id', 'LABEL0': 'return'}, inplace=True)

#     # preprocessing alpha data
#     print('merging alpha data ...')
#     alpha = pd.merge(features, labels, on=['id', 'date'], how='outer')
#     print('dropping nan stock return values ...')
#     alpha.dropna(subset=['return'], inplace=True)

#     alpha['id'] = alpha['id'].apply(lambda x: int(x[2:]))
#     alpha['date'] = alpha['date'].astype(str).apply(lambda x: ''.join(x.split('-'))).astype(np.int32)

#     if bench != 'allAshare':
#         member = read_member(member_dir, start_date, end_date)
#         alpha = pd.merge(member, alpha, on=['id', 'date'])

#     # store alpha data by dates
#     dateLst = np.sort(pd.unique(alpha['date']))
#     with trange(len(dateLst)) as date_bar:    
#         for i in date_bar:
#             date_i = dateLst[i]
#             date_bar.set_description(f'Saving alpha data from Qlib on trading date {date_i}')
#             alpha_i = alpha.loc[alpha['date'] == date_i]
#             alpha_i.to_csv(f'{save_dir}/Data/Qlib_Alpha/{bench}/{date_i}.csv', index=False)