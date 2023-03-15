import os
import sys
import math
import copy
import numpy as np
import pandas as pd
import pickle
import pickle5
import quantstats as qs
import scipy.stats as sp
from functools import reduce
from tqdm import trange


import warnings
warnings.filterwarnings("ignore")


D = {}

# Variables
BATCH_SIZE = 500
LEARNING_RATE = 1e-4
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-5
MAX_EPOCH = 1000
EPOCH = 100
TAU = 0.5


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
def read_return_data_vwap(stocklist_dir: str, save_dir: str, start_date: int, end_date: int, T: int, gap: bool):

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
            stockDF['vwap'] = stockDF['amount'] / stockDF['volume']
            if gap:
                stockDF['return'] = stockDF['vwap'].shift(-(T+1)) / stockDF['vwap'].shift(-1) - 1
            else:
                stockDF['return'] = stockDF['vwap'].shift(-T) / stockDF['vwap'] - 1
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


def load_dataset(feature_dir: str, trDays_dir: str, return_dir: str, price_dir: str, member_dir: str, start_year: int, end_year: int,
                 name: str, bench: str, n_stock: int):

    # create folder to store the factor data
    if not os.path.exists(f'{save_dir}/Data/{name}/{bench}'):
        os.makedirs(f'{save_dir}/Data/{name}/{bench}')

    # get the trading date sequence
    start_date, end_date = int(f'{start_year}0101'), int(f'{end_year}1231')
    trDate = read_pkl5_data(trDays_dir)
    trDate = trDate.index.to_numpy()
    trDate = trDate[(trDate >= start_date) & (trDate <= end_date)]

    if bench != 'allAshare':
        member_df = read_member(member_dir, trDate[0], trDate[-1])
        member_df['cnt'] = 1
        member_df = member_df.groupby('id')['cnt'].sum()
        member_df = member_df.to_frame()
        member_df.reset_index(drop=False, inplace=True)
    else:
        member_df = None

    return_df = pd.DataFrame()
    # load the training dataset
    with trange(len(trDate)) as date_bar:
        for i in date_bar:
            date_i = trDate[i]
            date_bar.set_description(f'Loading data on date {date_i}')

            # get data from different datasets
            return_i = pd.read_csv(f'{return_dir}/{date_i}.csv')
            return_df = return_df.append(return_i)

    price_df = pd.DataFrame()
    # load the training dataset
    with trange(len(trDate)) as date_bar:
        for i in date_bar:
            date_i = trDate[i]
            date_bar.set_description(f'Loading data on date {date_i}')

            # get data from different datasets
            price_i = pd.read_csv(f'{price_dir}/{date_i}.csv')
            price_df = price_df.append(price_i)

    # get the matrix of all feature data
    dfLst = []
    for feature_i in os.listdir(feature_dir):
        df_i = pd.DataFrame()
        with trange(len(trDate)) as date_bar:
            for i in date_bar:
                date_i = trDate[i]
                date_bar.set_description(f'Loading factor data {feature_i} from {name} on trading date {date_i}')
                factor_i = pd.read_csv(f'{feature_dir}/{feature_i}/allAshare/{date_i}.csv')
                df_i = df_i.append(factor_i)
        dfLst.append(df_i)

    feature_df = reduce(lambda df1, df2: pd.merge(df1, df2, on=['id', 'date'], how='inner'), dfLst)

    feature_cols = [col for col in feature_df.columns if not col in ['id', 'date']]
    market_lst = []
    for f in feature_cols:
        temp_feature = feature_df.groupby('date')[f].mean()
        temp_feature.name = f'{f}_mean'
        temp_feature = temp_feature.to_frame()
        temp_feature.reset_index(drop=False, inplace=True)
        market_lst.append(temp_feature)

    if not member_df is None:
        feature_id = pd.DataFrame({'id': pd.unique(feature_df['id'])})
        return_id = pd.DataFrame({'id': pd.unique(return_df['id'])})
        price_id = pd.DataFrame({'id': pd.unique(price_df['id'])})
        member_df = pd.merge(feature_id, member_df, on='id', how='inner')
        member_df = pd.merge(return_id, member_df, on='id', how='inner')
        member_df = pd.merge(price_id, member_df, on='id', how='inner')
        member_df.sort_values(by='cnt', inplace=True)
        member_df = member_df.iloc[-n_stock:,:]
        feature_df = pd.merge(feature_df, member_df, on='id', how='inner')
        del feature_df['cnt']

    feature_df.sort_values(by=['id', 'date'], inplace=True)
    feature_df.set_index(keys=['id', 'date'], inplace=True)
    return_df.set_index(keys=['id', 'date'], inplace=True)
    price_df.set_index(keys=['id', 'date'], inplace=True)

    df = pd.merge(feature_df, return_df, on=['id', 'date'], how='inner')
    df = pd.merge(df, price_df, on=['id', 'date'], how='inner')

    if member_df is None:
        write_pkl5_data(f'{save_dir}/Data/{name}/allAshare/{trDate[0]}_{trDate[-1]}.pkl', df)
    else:
        write_pkl5_data(f'{save_dir}/Data/{name}/{bench}/{trDate[0]}_{trDate[-1]}.pkl', df)


def preprocess(save_dir: str, start_date: int, end_date: int, bench: str, name: str):

    df = read_pkl5_data(f'{save_dir}/Data/{name}/{bench}/{start_date}_{end_date}.pkl')
    ret = df['return'].copy()
    price = df['close'].copy()
    del df['return'], df['close']
    featureLst = []
    for f in df.columns:
        print(f'Preprocessing feature {f} ...')
        temp = df.loc[:,f]
        temp = temp.unstack()
        features = copy.deepcopy(temp.values)
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
        temp.loc[:,:] = features
        temp = temp.stack()
        temp.name = f
        temp = temp.to_frame()
        featureLst.append(temp)

    df = reduce(lambda df1, df2: pd.merge(df1, df2, on=['id', 'date'], how='inner'), featureLst)
    df = pd.merge(df, ret, on=['id', 'date'])
    df = pd.merge(df, price, on=['id', 'date'])

    print(df)
    write_pkl5_data(f'{save_dir}/Data/{name}/{bench}/{start_date}_{end_date}_processed.pkl', df)

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


def get_metrics(save_dir: str, start_date: int, end_date: int, bench: str, name: str, weights: str, sign):

    # create folder to store the return data
    if not os.path.exists(f'{save_dir}/Metrics/{bench}'):
        os.makedirs(f'{save_dir}/Metrics/{bench}')

    df = read_pkl5_data(f'{save_dir}/Data/{name}/{bench}/{start_date}_{end_date}_processed.pkl')

    df.reset_index(drop=False, inplace=True)
    return_df = df.loc[:,['id', 'date', 'return']].copy()
    del df['return']

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

    elif weights == 'ICIR':

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
        featureIC_cols = [f'{col}_IC' for col in feature_cols]
        IC_df[featureIC_cols] = IC_df[featureIC_cols].rolling(252).mean() / IC_df[featureIC_cols].rolling(252).std()
        IC_df.dropna(subset=featureIC_cols, inplace=True)
        df = pd.merge(df, IC_df, on='date', how='inner')
        for f in feature_cols:
            df[f] *= df[f'{f}_IC']
        df['sum'] = df[featureIC_cols].abs().sum(axis=1)
        df[feature_cols] = df[feature_cols].div(df["sum"], axis=0)
        df['score'] = df[feature_cols].sum(axis=1)

    elif weights == 'weighted_ICIR':

        def get_WIC(x):
            return WIC(x.iloc[:,0], x.iloc[:,1])

        df = pd.merge(df, return_df, on=['id', 'date'])
        feature_cols = [col for col in df.columns if not col in ['id', 'date', 'return', 'close']]
        ICLst = []
        for i in range(len(feature_cols)):
            f = feature_cols[i]
            print(f'Calculate WIC of factor {f}')
            df[f] *= sign[i]
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

    df.to_csv(f'{save_dir}/Metrics/{bench}/{weights}_{start_date}_{end_date}.csv')


def back_test(save_dir: str, trDays_dir: str, start_date: int, end_date: int, bench: str, name: str, thres, method, weights):

    # create folder to store the return data
    if not os.path.exists(f'{save_dir}/Report/{bench}'):
        os.makedirs(f'{save_dir}/Report/{bench}')

    df = pd.read_csv(f'{save_dir}/Metrics/{bench}/{weights}_{start_date}_{end_date}.csv')
    del df['return']

    # get the trading date sequence
    trDate = read_pkl5_data(trDays_dir)
    trDate = trDate.index.to_numpy()

    MonthDF = df['date'].apply(lambda x: int(str(x)[:6]))
    MonthLst = np.sort(pd.unique(MonthDF))

    MonthFirstDate = []
    for m in MonthLst:
        month_i = trDate[trDate >= int(f'{m}01')][0]
        MonthFirstDate.append(month_i)

    df = df[df['date'].isin(MonthFirstDate)]
    df.sort_values(by=['id', 'date'], inplace=True)
    df['return'] = df.groupby(by='id')['close'].pct_change(1)
    df.dropna(subset=['return'], inplace=True)
    bench_df = df.loc[:,['id', 'date', 'return']].copy()
    bench_df = bench_df.groupby(by='date')['return'].mean()
    bench_df = bench_df.to_frame()
    bench_df.reset_index(drop=False, inplace=True)
    bench_df.rename(columns={'return': 'benchmark'}, inplace=True)

    def topN(x):
        score = x['score'].values
        ret = x['return'].values
        ids = np.argsort(score)[::-1]
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
    dateLst = np.sort(portfolio['date'])
    portfolio['date'] = portfolio['date'].apply(lambda date_i: pd.Timestamp(int(str(date_i)[:4]), int(str(date_i)[4:6]), int(str(date_i)[6:])))
    portfolio.set_index('date', inplace=True)
    print(portfolio)

    report_dir = f'{save_dir}/Report/{dateLst[0]}_{dateLst[-1]}_{bench}_{name}_{method}_{thres}_{weights}.html'

    qs.reports.html(portfolio['return'], portfolio['benchmark'],
        title=f'Report of long-short portfolio with factor predicted by {name}',
        output=report_dir)
    
    print('Report saved in %s' % (report_dir))