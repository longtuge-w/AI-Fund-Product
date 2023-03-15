import os
import sys
import math
import copy
import numpy as np
import pandas as pd
import pickle
import pickle5
import xgboost as xgb
import quantstats as qs
import scipy.stats as sp
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
from functools import reduce
from tqdm import trange
from joblib import dump, load
from DNN import DNN
from LSTM import LSTMModel
from GRU import GRUModel
from ALSTM import ALSTMModel
from TCN import TCNModel
from Transformer import Transformer


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

import warnings
warnings.filterwarnings("ignore")


D = {}

# Variables
BATCH_SIZE = 1024
LEARNING_RATE = 1e-4
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-5
MAX_EPOCH = 5
TAU = 0.5


class Data(Dataset):
    """
    The simple Dataset object from torch that can produce batchs of data
    """
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


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


def get_preprocess_stock(data):
    "data is M * F"
    data = np.array(data, dtype = np.float32)
    a = np.zeros((3, data.shape[-1]))
    t = np.nan_to_num(data, nan = np.nan, neginf = 1e9)
    a[0, :] = np.nanmin(t, axis = 0)
    t = np.nan_to_num(data, nan = np.nan, posinf = -1e9)
    a[2, :] = np.nanmax(t, axis = 0)
    for i in range(data.shape[-1]):
        data[:,i] = np.nan_to_num(data[:,i], nan = np.nan, posinf = a[2,i], neginf = a[0,i])
        try:
            data[:,i] = (data[:,i] - a[0,i]) / (a[2,i] - a[0,i])
        except:
            if i not in D.keys():
                D[i] = 0
            D[i] += 1
            print(i)
            print(data[:,i])
    for i in range(data.shape[-1]):
        nan_value = 0.0 if np.nanmean(data[:,i]) == np.nan else np.nanmean(data[:,i])
        data[:,i] = np.nan_to_num(data[:,i], nan = nan_value)
        a[1, i] = nan_value
    return data, a

# data: [date, stock, feature]
def get_preprocess(data):
    A = []
    for i in range(data.shape[1]):
        data[:,i,:], a = get_preprocess_stock(data[:,i,:])
        A.append(a)
    return data, A

def preprocess_stock(data, a):
    for i in range(data.shape[-1]):
        data[:,i] = np.nan_to_num(data[:,i], nan = a[1,i], posinf = a[2,i], neginf = a[0,i])
    for i in range(data.shape[0]):
        a[0,:] = np.minimum(a[0,:], data[i,:])
        a[2,:] = np.maximum(a[2,:], data[i,:])
        for j in range(data.shape[-1]):
            try:
                data[i,j] = (data[i,j] - a[0,j]) / (a[2,j] - a[0,j])
            except:
                print("!!!!!!/n/n")
                print(i,j)
    return data

def preprocess(data, A):
    for i in range(data.shape[1]):
        data[:,i,:] = preprocess_stock(data[:,i,:], A[i])
    return data


# read the stock data
def read_return_data(stocklist_dir: str, save_dir: str, start_date: int, end_date: int, T: int, gap: bool):

    # create the folder if exists
    if gap:
        if not os.path.exists(f'{save_dir}/Data/Return/T+1_T+{T+1}'):
            os.makedirs(f'{save_dir}/Data/Return/T+1_T+{T+1}')
    else:
        if not os.path.exists(f'{save_dir}/Data/Return/T_T+{T}'):
            os.makedirs(f'{save_dir}/Data/Return/T_T+{T}')

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
                stock_i.to_csv(f'{save_dir}/Data/Return/T+1_T+{T+1}/{date_i}.csv', index=False)
            else:
                stock_i.to_csv(f'{save_dir}/Data/Return/T+1_T+{T+1}/{date_i}.csv', index=False)

# read the stock data
def read_price_data(stocklist_dir: str, member_dir: str, save_dir: str, start_date: int, end_date: int, T: int):

    # create the folder if exists
    if not os.path.exists(f'{save_dir}/Data/Price/{T}'):
        os.makedirs(f'{save_dir}/Data/Price/{T}')

    member_dir

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
            stockDF = stockDF.loc[(stockDF['date'] >= start_date) & (stockDF['date'] <= end_date)]
            stockDF = stockDF.loc[:,['id', 'date', 'close']]
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
            stock_i.to_csv(f'{save_dir}/Data/Return/T_T+{T}/{date_i}.csv', index=False)

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
            start_date_i = max(exist_date, start_date)
        df_i = df_i.stack().reset_index()
        df_i.rename(columns={'level_0': 'date', 'level_1': 'id', 0: feature_i}, inplace=True)
        df_i = df_i.loc[df_i['date'] >= start_date_i]
        df_i['id'] = df_i['id'].apply(lambda x: x[:-3])
        df_i = df_i.loc[df_i['id'].apply(lambda x: x.isdigit())]
        df_i['id'] = df_i['id'].astype(np.int32)
        # get the member of stock index
        if bench != 'allAshare':
            member = read_member(member_dir, start_date_i, end_date)
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


def random_batch_mlp(x, y):
    ind = np.random.randint(0, len(x), BATCH_SIZE)
    batch_x, batch_y = x[ind].astype(np.float), y[ind].astype(np.float)
    return batch_x, batch_y

# pearson correlation
def pearson_corr(x, y):
    vx = x - x.mean()
    vy = y - y.mean()
    cost = (vx * vy).sum() / ((vx ** 2).sum().sqrt() * (vy ** 2).sum().sqrt())
    return cost

# this loss is the sum of negative value of IC over each stock
def IC_loss(logits, target):
    return -pearson_corr(logits, target)

# weighted IC loss function
def WIC_loss(logits, target):
    rank = (logits * 1).argsort().argsort()
    weights = 0.5 ** (rank / (len(rank)-1))
    mean_w_x = (weights*logits).sum()
    mean_w_r = (weights*target).sum()
    numera = (weights*logits*target).sum() - mean_w_x * mean_w_r
    var_w_x = (weights*logits**2).sum() - mean_w_x ** 2
    var_w_r = (weights*target**2).sum() - mean_w_r ** 2
    denomi = math.sqrt(abs(var_w_x * var_w_r))
    return -numera / denomi

# CCC loss function, a combination of MSE loss and IC loss
def CCC_loss(logits, target):
    logits_mean, target_mean = logits.mean(), target.mean()
    logits_var, target_var = ((logits - logits_mean) ** 2).sum(), ((target - target_mean) ** 2).sum()
    denomi = torch.sum((logits - logits_mean) * (target - target_mean))
    ccc = 2 * denomi / (logits_var + target_var + (logits_mean - target_mean) ** 2)
    return -ccc

# weighted ccc loss function
def WCCC_loss(logits, target):
    weight = torch.argsort(torch.argsort(target))
    weight = weight / torch.max(weight)
    weight = torch.exp((1 - weight) * torch.log(torch.tensor(0.5)) / TAU)
    weight = weight / torch.sum(weight)
    logits_mean, target_mean = (weight * logits).sum(), (weight * target).sum()
    logits_var, target_var = (weight * (logits - logits_mean) ** 2).sum(), (weight * (target - target_mean) ** 2).sum()
    denomi = torch.sum(weight * logits * target) - torch.sum(weight * logits) * torch.sum(weight * target)
    ccc = 2 * denomi / (logits_var + target_var + (logits_mean - target_mean) ** 2)
    return -ccc


# load the features, label, and information of different datasets
def load_dataset(save_dir: str, dates: np.array, return_df: pd.DataFrame, bench: str, factor: str):

    factor_, label_, id_, tdate_ = [], [], [], []

    # load the training dataset
    with trange(len(dates)) as date_bar:
        for i in date_bar:
            date_i = dates[i]
            date_bar.set_description(f'Loading data on date {date_i}')

            # get data from different datasets
            data_i = []

            if not factor is None:
                if not os.path.exists(f'{save_dir}/Unit/Factor/{factor}/{bench}/{date_i}.pkl'):
                    continue
                factor2_i = read_pkl5_data(f'{save_dir}/Unit/Factor/{factor}/{bench}/{date_i}.pkl')
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

            factorlst_i = []

            if not factor is None:
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
                    factorlst_i.append(features)
                factor_i = np.stack(factorlst_i, axis=2)
                factor_.append(factor_i)

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

    if not factor is None:
        factor_ = np.concatenate(factor_, axis=0)
        
    label_ = np.concatenate(label_, axis=0)
    id_ = np.concatenate(id_, axis=0)
    tdate_ = np.concatenate(tdate_, axis=0)

    return factor_, label_, id_, tdate_

# save the feature matrix data for further use
def get_rolling_feature_data(save_dir: str, feature_dir: str, trDays_dir: str, return_dir: str, member_dir: str, lgbm, NN, loss: str, 
                start_year: int, end_year: int, train_window: int, eval_window: int, test_window: int, bench: str, name: str, n_stock: int,
                save_npy: bool=True):

    # create folder to store the factor data
    if not os.path.exists(f'{save_dir}/Feature/{name}/{bench}'):
        os.makedirs(f'{save_dir}/Feature/{name}/{bench}')

    if not os.path.exists(f'{save_dir}/Model/{bench}'):
        os.makedirs(f'{save_dir}/Model/{bench}')

    # get the trading date sequence
    start_date, end_date = int(f'{start_year}0101'), int(f'{end_year}1231')
    timeLst = read_pkl5_data(trDays_dir)
    timeLst = timeLst.index.to_numpy()
    timeLst = timeLst[(timeLst >= start_date) & (timeLst <= end_date)]

    # return_df = pd.DataFrame()
    # # load the training dataset
    # with trange(len(timeLst)) as date_bar:
    #     for i in date_bar:
    #         date_i = timeLst[i]
    #         date_bar.set_description(f'Loading data on date {date_i}')

    #         # get data from different datasets
    #         return_i = pd.read_csv(f'{return_dir}/{date_i}.csv')
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
            if os.path.exists(f'{feature_dir}/{train_dates[0]}_{train_dates[-1]}_factor1.npy'):
                train_feature = np.load(f'{feature_dir}/{train_dates[0]}_{train_dates[-1]}_factor1.npy', allow_pickle=True)
                train_Y = np.load(f'{feature_dir}/{train_dates[0]}_{train_dates[-1]}_label.npy', allow_pickle=True)
                train_id = np.load(f'{feature_dir}/{train_dates[0]}_{train_dates[-1]}_id.npy', allow_pickle=True)
                train_tdate = np.load(f'{feature_dir}/{train_dates[0]}_{train_dates[-1]}_date.npy', allow_pickle=True)
            else:
                train_feature, train_Y, train_id, train_tdate = load_dataset(save_dir, train_dates, return_df, bench, name)
                np.save(f'{feature_dir}/{train_dates[0]}_{train_dates[-1]}_factor1.npy', train_feature, allow_pickle=True)
                np.save(f'{feature_dir}/{train_dates[0]}_{train_dates[-1]}_label.npy', train_Y, allow_pickle=True)
                np.save(f'{feature_dir}/{train_dates[0]}_{train_dates[-1]}_id.npy', train_id, allow_pickle=True)
                np.save(f'{feature_dir}/{train_dates[0]}_{train_dates[-1]}_date.npy', train_tdate, allow_pickle=True)
        else:
            train_feature, train_Y, train_id, train_tdate = load_dataset(save_dir, train_dates, return_df, bench, name)

        train_Y = np.nan_to_num(train_Y, nan=0)

        if save_npy:
            if os.path.exists(f'{feature_dir}/{test_dates[0]}_{test_dates[-1]}_factor1.npy'):
                test_factor = np.load(f'{feature_dir}/{test_dates[0]}_{test_dates[-1]}_factor1.npy', allow_pickle=True)
                test_Y = np.load(f'{feature_dir}/{test_dates[0]}_{test_dates[-1]}_label.npy', allow_pickle=True)
                test_id = np.load(f'{feature_dir}/{test_dates[0]}_{test_dates[-1]}_id.npy', allow_pickle=True)
                test_tdate = np.load(f'{feature_dir}/{test_dates[0]}_{test_dates[-1]}_date.npy', allow_pickle=True)
            else:
                test_factor, test_Y, test_id, test_tdate = load_dataset(save_dir, test_dates, return_df, bench, name)
                np.save(f'{feature_dir}/{test_dates[0]}_{test_dates[-1]}_factor1.npy', test_factor, allow_pickle=True)
                np.save(f'{feature_dir}/{test_dates[0]}_{test_dates[-1]}_label.npy', test_Y, allow_pickle=True)
                np.save(f'{feature_dir}/{test_dates[0]}_{test_dates[-1]}_id.npy', test_id, allow_pickle=True)
                np.save(f'{feature_dir}/{test_dates[0]}_{test_dates[-1]}_date.npy', test_tdate, allow_pickle=True)
        else:
            test_factor, test_Y, test_id, test_tdate = load_dataset(save_dir, test_dates, return_df, bench, name)

        test_Y = np.nan_to_num(test_Y, nan=0)

        # load the evaluation data set if it exists
        if len(eval_dates) != 0:
            if save_npy:
                if os.path.exists(f'{feature_dir}/{eval_dates[0]}_{eval_dates[-1]}_factor1.npy'):
                    eval_factor = np.load(f'{feature_dir}/{eval_dates[0]}_{eval_dates[-1]}_factor1.npy', allow_pickle=True)
                    eval_Y = np.load(f'{feature_dir}/{eval_dates[0]}_{eval_dates[-1]}_label.npy', allow_pickle=True)
                    eval_id = np.load(f'{feature_dir}/{eval_dates[0]}_{eval_dates[-1]}_id.npy', allow_pickle=True)
                    eval_tdate = np.load(f'{feature_dir}/{eval_dates[0]}_{eval_dates[-1]}_date.npy', allow_pickle=True)
                else:
                    eval_factor, eval_Y, eval_id, eval_tdate = load_dataset(save_dir, eval_dates, return_df, bench, name)
                    np.save(f'{feature_dir}/{eval_dates[0]}_{eval_dates[-1]}_factor1.npy', eval_factor, allow_pickle=True)
                    np.save(f'{feature_dir}/{eval_dates[0]}_{eval_dates[-1]}_label.npy', eval_Y, allow_pickle=True)
                    np.save(f'{feature_dir}/{eval_dates[0]}_{eval_dates[-1]}_id.npy', eval_id, allow_pickle=True)
                    np.save(f'{feature_dir}/{eval_dates[0]}_{eval_dates[-1]}_date.npy', eval_tdate, allow_pickle=True)
            else:
                eval_factor, eval_Y, eval_id, eval_tdate = load_dataset(save_dir, eval_dates, return_df, bench, name)

            eval_Y = np.nan_to_num(eval_Y, nan=0)
        
        train_return = np.nan_to_num(train_Y, nan=0)
        n_features = train_feature.shape[-1]

        mask = ~(train_feature == 0).all(axis=(1, 2))
        train_feature = train_feature[mask]
        train_return = train_return[mask]

        train_feature = np.nan_to_num(train_feature, nan=0)
        train_feature = torch.from_numpy(train_feature)
        train_feature = train_feature.to(torch.float32)
        train_factor_dataset = Data(train_feature, train_Y)
        train_factor_dataloader = DataLoader(train_factor_dataset, BATCH_SIZE, shuffle=False)

        # train the model one by one
        if os.path.exists(f'{save_dir}/Model/{bench}/{NN}_{loss}_factor1_{year_i}_{year_i+train_window-1}.m'):
            print(f'Model {NN} for factor 1 already exists, loading model ...')
        else:
            # choose the right model to train
            if NN == 'DNN':
                model = DNN(n_features=n_features, hidden_size=64, date_size=60)
            elif NN == 'LSTM':
                model = LSTMModel(input_size=n_features, output_size=1)
            elif NN == 'GRU':
                model = GRUModel(input_size=n_features, output_size=1)
            elif NN == 'ALSTM':
                model = ALSTMModel(input_size=n_features, output_size=1)
            elif NN == 'TCN':
                model = TCNModel(num_input=n_features, output_size=1)
            elif NN == 'Transformer':
                model = Transformer(input_size=n_features, output_size=1)
            elif NN == None:
                model = None
            else:
                raise ValueError(f'The paramter NN should be DNN/LSTM/GRU/ALSTM/TCN/Transformer/None, get {NN} instead')
        
            if loss == 'MSE':
                loss_func = nn.MSELoss()
            elif loss == 'IC':
                loss_func = IC_loss
            elif loss == 'WIC':
                loss_func = WIC_loss
            elif loss == 'CCC':
                loss_func = CCC_loss
            elif loss == 'WCCC':
                loss_func = WCCC_loss
            else:
                raise ValueError(f'The parameter loss should be MSE/IC/CCC/WCCC, get {loss} instead')

            if not model is None:
                best_model, _ = train(model, train_factor_dataloader, loss_func)
                dump(best_model, f'{save_dir}/Model/{bench}/{NN}_{loss}_{year_i}_{year_i+train_window-1}.m')

        # if not os.path.exists(f'{save_dir}/Model/{bench}/XGB_{year_i}_{year_i+train_window-1}.joblib') and lgbm:
        #     train_feature = train_feature.reshape((-1,train_feature.shape[-1]))
        #     train_return = train_return.reshape(-1)
        #     model = xgb.XGBRegressor(eta=5e-2)
        #     model = model.fit(train_feature, train_return)
        #     dump(model, f'{save_dir}/Model/{bench}/XGB_{year_i}_{year_i+train_window-1}.joblib')


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
            with torch.no_grad():
                y_pred = model(x_test)

            # calculate loss
            Loss += criterion(y_pred, y_test).item()

            # set information for the bar
            test_bar.set_postfix(evaluate_loss=Loss / (i+1))

            # delete data to release memory
            del x_test, y_test
            torch.cuda.empty_cache()

        return Loss / (i+1)


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
        min_loss = math.inf
        best_model = None

        # set the bar to check the progress
        with trange(train_data_size) as train_bar:
            for i in train_bar:
                train_bar.set_description(f'Training batch {i+1}')
                x_train, y_train = next(train_dataiter)
                x_train, y_train = x_train.float(), y_train.float()
                x_train, y_train = x_train.to(device), y_train.to(device)

                # clear the gradients
                optimizer.zero_grad()

                # compute the model output
                y_pred = model(x_train)

                # calculate loss
                train_loss = criterion(y_pred, y_train)

                Total_loss += train_loss.item()

                # credit assignment
                train_loss.backward()

                # update model weights
                optimizer.step()

                # set information for the bar
                train_bar.set_postfix(train_loss=Total_loss / (i+1))

                # delete data to release memory
                del x_train, y_train
                torch.cuda.empty_cache()

            # see whether the trained model after this epoch is the currently best
            if valid_dataloader is not None:
                model.eval()
                loss = evaluate(model, valid_dataloader, criterion)
                if loss < min_loss:
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
            with torch.no_grad():
                y_pred = model(x_test)
                factors.append(y_pred)

            # delete data to release memory
            del x_test, y_test
            torch.cuda.empty_cache()

    # concatenate the data to get the output factors
    factors = torch.cat(factors, dim=0).cpu().detach().numpy()

    return factors


def get_predict(save_dir: str, feature_dir: str, trDays_dir: str, NN, loss: str, 
                start_year: int, end_year: int, train_window: int, eval_window: int, test_window: int, bench: str):
    
    # create folder to store the factor data
    if not os.path.exists(f'{save_dir}/Factor/{bench}'):
        os.makedirs(f'{save_dir}/Factor/{bench}')

    # get the trading date sequence
    start_date, end_date = int(f'{start_year}0101'), int(f'{end_year}1231')
    timeLst = read_pkl5_data(trDays_dir)
    timeLst = timeLst.index.to_numpy()
    timeLst = timeLst[(timeLst >= start_date) & (timeLst <= end_date)]

    factor = pd.DataFrame()

    # iterately train and test the model
    for year_i in range(start_year, end_year-train_window-eval_window-test_window+2):

        model = load(f'{save_dir}/Model/{bench}/{NN}_{loss}_{year_i}_{year_i+train_window-1}.m')

        test_dates = timeLst[(int(f'{year_i+train_window+eval_window}0101') <= timeLst) &
             (timeLst <= int(f'{year_i+train_window+eval_window+test_window-1}1231'))]
        
        test_factor = np.load(f'{feature_dir}/{test_dates[0]}_{test_dates[-1]}_factor1.npy', allow_pickle=True)
        test_Y = np.load(f'{feature_dir}/{test_dates[0]}_{test_dates[-1]}_label.npy', allow_pickle=True)
        test_id = np.load(f'{feature_dir}/{test_dates[0]}_{test_dates[-1]}_id.npy', allow_pickle=True)
        test_tdate = np.load(f'{feature_dir}/{test_dates[0]}_{test_dates[-1]}_date.npy', allow_pickle=True)

        test_Y = np.nan_to_num(test_Y, nan=0)

        test_factor = np.nan_to_num(test_factor, nan=0)
        test_factor = torch.from_numpy(test_factor)
        test_factor = test_factor.to(torch.float32)
        test_factor_dataset = Data(test_factor, test_Y)
        test_factor_dataloader = DataLoader(test_factor_dataset, BATCH_SIZE, shuffle=False)

        test_score = get_factor(model, test_factor_dataloader)
        test_df = pd.DataFrame({'id': test_id, 'date': test_tdate, 'return': test_Y, 'score': test_score})
        factor = factor.append(test_df)

    dateLst = np.sort(pd.unique(factor['date']))
    start, end = dateLst[0], dateLst[-1]
    factor.to_csv(f'{save_dir}/Factor/{bench}/{NN}_{loss}_{start}_{end}.csv')


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
def backtest(save_dir: str, start_date: int, end_date: int, bench: str, T: int, NN, loss: str,
             method: str, thres, sign: int=1, gap: bool=True):

    # create folder to store the return data
    if not os.path.exists(f'{save_dir}/Report/{bench}/{T}'):
        os.makedirs(f'{save_dir}/Report/{bench}/{T}')

    df = pd.read_csv(f'{save_dir}/Factor/{bench}/{NN}_{loss}_{start_date}_{end_date}.csv')
    dates = np.sort(pd.unique(df['date']))
    
    if gap:
        dates = [dates[i] for i in range(0,len(dates),T+1)]
    else:
        dates = [dates[i] for i in range(0,len(dates),T)]

    df = df[df['date'].isin(dates)]
    return_df = df.loc[:,['id', 'date', 'return']].copy()
    df = df.loc[:,['date', 'score', 'return']].copy()

    bench_df = return_df.groupby(by='date')['return'].mean().reset_index()
    bench_df.rename(columns={'return': 'benchmark'}, inplace=True)

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
        report_dir = f'{save_dir}/Report/{bench}/{T}/{NN}_{loss}_{start_date}_{end_date}_T+1_T+{T+1}.html'
    else:
        report_dir = f'{save_dir}/Report/{bench}/{T}/{NN}_{loss}_{start_date}_{end_date}_T_T+{T}.html'

    qs.reports.html(portfolio['return'], portfolio['benchmark'],
        title=f'Report of long portfolio with factor predicted by {NN} and {loss}',
        output=report_dir)
    
    print('Report saved in %s' % (report_dir))