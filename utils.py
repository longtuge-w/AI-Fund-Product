import os
import sys
import copy
import pickle5
import numpy as np
import pandas as pd
import xgboost as xgb
import scipy.stats as sp
from scipy.special import logsumexp
from tqdm import trange
from joblib import load, dump
from functools import reduce


N_STOCKS = 500
DEFAULT_EPS = 1e-10
PADDED_Y_VALUE = -1
LAMBDA = 1e-2


# def gradient(predt: np.ndarray, dtrain: xgb.DMatrix) -> np.ndarray:
#     '''Compute the gradient squared log error.'''
#     y = dtrain.get_label()
#     return (np.log1p(predt) - np.log1p(y)) / (predt + 1)

# def hessian(predt: np.ndarray, dtrain: xgb.DMatrix) -> np.ndarray:
#     '''Compute the hessian for squared log error.'''
#     y = dtrain.get_label()
#     return ((-np.log1p(predt) + np.log1p(y) + 1) /
#             np.power(predt + 1, 2))

# def squared_log(predt: np.ndarray,
#                 dtrain: xgb.DMatrix) -> Tuple[np.ndarray, np.ndarray]:
#     '''Squared Log Error objective. A simplified version for RMSLE used as
#     objective function.
#     '''
#     predt[predt < -1] = -1 + 1e-6
#     grad = gradient(predt, dtrain)
#     hess = hessian(predt, dtrain)
#     return grad, hess



def listmle_loss(y_pred, dtrain):
    """
    Compute the ListMLE loss and its gradient and Hessian.

    Parameters:
    y_true: np.ndarray, shape (n_items,)
        The true relevance labels for each item.
    y_pred: np.ndarray, shape (n_items,)
        The predicted relevance scores for each item.

    Returns:
    loss: float
        The ListMLE loss.
    grad: np.ndarray, shape (n_items,)
        The gradient of the ListMLE loss.
    hess: np.ndarray, shape (n_items, n_items)
        The Hessian of the ListMLE loss.
    """

    y_true = dtrain.get_label()
    n_items = y_true.shape[0]
    y_true = y_true.astype(np.float16)
    y_pred = y_pred.astype(np.float16)

    # Sort the true labels and the predicted scores in descending order
    idx = np.argsort(-y_true)
    y_true = y_true[idx]
    y_pred = y_pred[idx]

    # Compute the ListMLE loss and its gradient and Hessian
    loss = 0.0
    grad = np.zeros(n_items)
    hess = np.zeros((n_items, n_items))
    for i in range(n_items):
        exp_sum = np.sum(np.exp(y_pred[i:]))
        loss_i = np.log(exp_sum) - y_pred[i] + y_true[i] * y_pred[i] - y_true[i] * np.log(exp_sum)
        loss += loss_i

        grad_i = y_true[i] - np.sum(np.exp(y_pred[i:]) / exp_sum)
        grad[i] = grad_i

        for j in range(i, n_items):
            hess_ij = - np.exp(y_pred[i]) * np.exp(y_pred[j]) / exp_sum**2
            hess[i, j] = hess_ij
            hess[j, i] = hess_ij

    return loss, grad, hess


def listnet_loss(y_pred, dtrain, lambda_=LAMBDA):
    """
    Compute the ListNet loss and its gradient and Hessian.

    Parameters:
    y_true: np.ndarray, shape (n_items,)
        The true relevance labels for each item.
    y_pred: np.ndarray, shape (n_items,)
        The predicted relevance scores for each item.
    lambda_=LAMBDA: float
        The regularization parameter.

    Returns:
    loss: float
        The ListNet loss.
    grad: np.ndarray, shape (n_items,)
        The gradient of the ListNet loss.
    hess: np.ndarray, shape (n_items, n_items)
        The Hessian of the ListNet loss.
    """

    y_true = dtrain.get_label()
    n_items = y_true.shape[0]
    y_true = y_true.astype(np.float16)
    y_pred = y_pred.astype(np.float16)

    # Compute the softmax function of the predicted scores
    softmax = np.exp(y_pred) / np.sum(np.exp(y_pred))

    # Compute the ListNet loss and its gradient and Hessian
    loss = 0.0
    grad = np.zeros(n_items, dtype=np.float16)
    hess = np.zeros((n_items, n_items), dtype=np.float16)
    for j in range(n_items):
        loss_j = - y_true[j] * np.log(softmax[j] + 1e-12)
        loss += loss_j

        grad_j = - y_true[j] / (softmax[j] + 1e-12)
        grad[j] = grad_j

        hess_jj = y_true[j] / (softmax[j] + 1e-12)**2
        hess[j, j] = hess_jj

        for k in range(n_items):
            if j == k:
                continue
            hess_jk = - y_true[j] * softmax[k] / (softmax[j] + 1e-12)**2
            hess[j, k] = hess_jk

    # Add the regularization term to the ListNet loss and its gradient and Hessian
    reg = lambda_ / 2 * np.sum(y_pred**2)
    loss += reg
    grad += lambda_ * y_pred
    hess += lambda_ * np.eye(n_items)

    return loss, grad, hess



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

def load_dataset(feature_dir: str, dates: np.array, return_df: pd.DataFrame, name: str, member_df: pd.DataFrame, n_stock: int):

    member_df = member_df.copy()
    return_df = return_df.copy()
    # get the matrix of all feature data
    dfLst = []
    for feature_i in os.listdir(feature_dir):
        df_i = pd.DataFrame()
        with trange(len(dates)) as date_bar:
            for i in date_bar:
                date_i = dates[i]
                date_bar.set_description(f'Loading factor data {feature_i} from {name} on trading date {date_i}')
                factor_i = pd.read_csv(f'{feature_dir}/{feature_i}/allAshare/{date_i}.csv')
                df_i = df_i.append(factor_i)
        dfLst.append(df_i)

    feature_df = reduce(lambda df1, df2: pd.merge(df1, df2, on=['id', 'date'], how='inner'), dfLst)

    if not member_df is None:
        feature_id = pd.DataFrame({'id': pd.unique(feature_df['id'])})
        return_id = pd.DataFrame({'id': pd.unique(return_df['id'])})
        member_df = pd.merge(feature_id, member_df, on='id', how='inner')
        member_df = pd.merge(return_id, member_df, on='id', how='inner')
        member_df.sort_values(by='cnt', inplace=True)
        member_df = member_df.iloc[-n_stock:,:]
        feature_df = pd.merge(feature_df, member_df, on='id', how='inner')
        del feature_df['cnt']

    feature_df.sort_values(by=['id', 'date'], inplace=True)
    feature_df.set_index(keys=['id', 'date'], inplace=True)
    return_df.set_index(keys=['id', 'date'], inplace=True)

    df = pd.merge(feature_df, return_df, on=['id', 'date'], how='inner')

    # stack all the feature data
    fLst = []
    for col in feature_df.columns:
        temp = df[col]
        # temp = temp.unstack()
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
        # temp.fillna(0, inplace=True)
        fLst.append(features)

    # save all the np array data for further use
    feature_ = np.stack(fLst, axis=2)
    feature_ = np.transpose(feature_, (1, 0, 2))
    df.reset_index(drop=False, inplace=True)
    id_ = df['id'].values
    date_ = df['date'].values
    return_ = df['return'].values

    return feature_, id_, date_, return_

# save the feature matrix data for further use
def get_rolling_feature_data(save_dir: str, feature_dir: str, data_dir: str, trDays_dir: str, return_dir: str, member_dir: str, 
                start_year: int, end_year: int, train_window: int, test_window: int, loss: str, bench: str, name: str, n_stock: int):

    if not os.path.exists(f'{save_dir}/Model/{bench}'):
        os.makedirs(f'{save_dir}/Model/{bench}')

    # get the trading date sequence
    start_date, end_date = int(f'{start_year}0101'), int(f'{end_year}1231')
    trDate = read_pkl5_data(trDays_dir)
    trDate = trDate.index.to_numpy()
    trDate = trDate[(trDate >= start_date) & (trDate <= end_date)]

    return_df = pd.DataFrame()
    # load the training dataset
    with trange(len(trDate)) as date_bar:
        for i in date_bar:
            date_i = trDate[i]
            date_bar.set_description(f'Loading data on date {date_i}')

            # get data from different datasets
            return_i = pd.read_csv(f'{return_dir}/{date_i}.csv')
            return_df = return_df.append(return_i)

    # iterately train and test the model
    for year_i in range(start_year, end_year-train_window-test_window+2):

        print(f'''processing data with training period {year_i}-{year_i+train_window-1}
        and testing period {year_i+train_window}-{year_i+train_window+test_window-1}''')

        # get the dates for corresponding data sets
        train_dates = trDate[(int(f'{year_i}0101') <= trDate) & (trDate <= int(f'{year_i+train_window-1}1231'))]
        test_dates = trDate[(int(f'{year_i+train_window}0101') <= trDate) 
                            & (trDate <= int(f'{year_i+train_window+test_window-1}1231'))]
        
        if bench != 'allAshare':
            member_df = read_member(member_dir, train_dates[0], test_dates[-1])
            member_df['cnt'] = 1
            member_df = member_df.groupby('id')['cnt'].sum()
            member_df = member_df.to_frame()
            member_df.reset_index(drop=False, inplace=True)
        else:
            member_df = None

        if os.path.exists(f'{data_dir}/{train_dates[0]}_{train_dates[-1]}_data.npy'):
            train_feature, train_id, train_date, train_return = np.load(f'{data_dir}/{train_dates[0]}_{train_dates[-1]}_data.npy', allow_pickle=True)
        else:
            train_feature, train_id, train_date, train_return = load_dataset(feature_dir, train_dates, return_df, name, member_df, n_stock)
            # train_feature, a = get_preprocess(train_feature)
            train_data = np.empty(4, dtype=object)
            train_data[:] = [train_feature, train_id, train_date, train_return]
            # save the feature data for further use
            np.save(f'{data_dir}/{train_dates[0]}_{train_dates[-1]}_data.npy', train_data, allow_pickle=True)

        if os.path.exists(f'{data_dir}/{test_dates[0]}_{test_dates[-1]}_data.npy'):
            test_feature, test_id, test_date, test_return = np.load(f'{data_dir}/{test_dates[0]}_{test_dates[-1]}_data.npy', allow_pickle=True)
        else:
            test_feature, test_id, test_date, test_return = load_dataset(feature_dir, test_dates, return_df, name, member_df, n_stock)
            # test_feature = preprocess(test_feature, a)
            test_data = np.empty(4, dtype=object)
            test_data[:] = [test_feature, test_id, test_date, test_return]
            # save the feature data for further use
            np.save(f'{data_dir}/{test_dates[0]}_{test_dates[-1]}_data.npy', test_data, allow_pickle=True)

        train_feature, test_feature = np.nan_to_num(train_feature, nan=0), np.nan_to_num(test_feature, nan=0)

        train_return = pd.DataFrame({'id': train_id, 'date': train_date, 'return': train_return})
        train_return.set_index(['id', 'date'], inplace=True)
        train_return = train_return.unstack().values
        
        train_return = np.nan_to_num(train_return, nan=0)
        train_return = np.transpose(train_return, (1, 0))

        mask = ~(train_feature == 0).all(axis=(1, 2))
        train_feature = train_feature[mask]
        train_return = train_return[mask]

        n_features = train_feature.shape[-1]
        train_feature = train_feature.reshape((-1, n_features))
        train_return = train_return.reshape(-1)

        if loss == 'ListMLE':
            obj = listmle_loss
            # metric = listmle_metrics
        elif loss == 'ListNet':
            obj = listnet_loss
            # metric = listnet_metric
        else:
            raise ValueError(f'The parameter loss should be ListMLE/ListNet, get {loss} instead')

        print(train_feature.shape)
        print(train_return.shape)

        if not os.path.exists(f'{save_dir}/Model/{bench}/XGB_{loss}_{year_i}_{year_i+train_window-1}.joblib'):
            dtrain = xgb.DMatrix(train_feature, train_return)
            model = xgb.train({}, dtrain=dtrain, num_boost_round=10, obj=obj)
            dump(model, f'{save_dir}/Model/{bench}/XGB_{loss}_{year_i}_{year_i+train_window-1}.joblib')
            sys.exit()