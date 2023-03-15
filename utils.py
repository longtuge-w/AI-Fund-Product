import os
import sys
import math
import copy
import numpy as np
import pandas as pd
import pickle
import pickle5
import joblib
from itertools import product
import quantstats as qs
import scipy.stats as sp
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from functools import reduce
from tqdm import trange
from DNN import CMLE
from LSTM import LSTMModel
from GRU import GRUModel
from ALSTM import ALSTMModel
from Transformer import make_transformer
from LTR import make_ltr_model


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Variables
BATCH_SIZE = 1024
LEARNING_RATE = 1e-4
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-5
MAX_EPOCH = 1000
EPOCH = 100
RANDOM_STATE = 1
N_STOCKS = 500
DEFAULT_EPS = 1e-10
PADDED_Y_VALUE = -1

D = {}


class ListMLE(nn.Module):
    """
    ListMLE loss introduced in "Listwise Approach to Learning to Rank - Theory and Algorithm".
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param eps: epsilon value, used for numerical stability
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: loss value, a torch.Tensor
    """
    def __init__(self):
        super(ListMLE, self).__init__()

    def forward(self, y_pred, y_true):
        # shuffle for randomised tie resolution
        random_indices = torch.randperm(y_pred.shape[-1])
        y_pred_shuffled = y_pred[:, random_indices]
        y_true_shuffled = y_true[:, random_indices]

        y_true_sorted, indices = y_true_shuffled.sort(descending=True, dim=-1)

        mask = y_true_sorted == PADDED_Y_VALUE

        preds_sorted_by_true = torch.gather(y_pred_shuffled, dim=1, index=indices)
        preds_sorted_by_true[mask] = float("-inf")

        max_pred_values, _ = preds_sorted_by_true.max(dim=1, keepdim=True)

        preds_sorted_by_true_minus_max = preds_sorted_by_true - max_pred_values

        cumsums = torch.cumsum(preds_sorted_by_true_minus_max.exp().flip(dims=[1]), dim=1).flip(dims=[1])

        observation_loss = torch.log(cumsums + DEFAULT_EPS) - preds_sorted_by_true_minus_max

        observation_loss[mask] = 0.0

        return torch.mean(torch.sum(observation_loss, dim=1))
    

class ListNet(nn.Module):
    """
    ListNet loss introduced in "Learning to Rank: From Pairwise Approach to Listwise Approach".
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param eps: epsilon value, used for numerical stability
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: loss value, a torch.Tensor
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
        preds_log = torch.log(preds_smax)

        return torch.mean(-torch.sum(true_smax * preds_log, dim=1))
    

class binary_listNet(nn.Module):
    """
    ListNet loss variant for binary ground truth data introduced in "Learning to Rank: From Pairwise Approach to Listwise Approach".
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param eps: epsilon value, used for numerical stability
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: loss value, a torch.Tensor
    """
    def __init__(self, eps=DEFAULT_EPS, padded_value_indicator=PADDED_Y_VALUE) -> None:
        super(binary_listNet, self).__init__()
        self.eps = eps
        self.padded_value_indicator = padded_value_indicator

    def forward(self, y_pred, y_true):

        y_pred = y_pred.clone()
        y_true = y_true.clone()

        mask = y_true == self.padded_value_indicator
        y_pred[mask] = float('-inf')
        y_true[mask] = 0.0
        normalizer = torch.unsqueeze(y_true.sum(dim=-1), 1)
        normalizer[normalizer == 0.0] = 1.0
        normalizer = normalizer.expand(-1, y_true.shape[1])
        y_true = torch.div(y_true, normalizer)

        preds_smax = F.softmax(y_pred, dim=1)

        preds_smax = preds_smax + self.eps
        preds_log = torch.log(preds_smax)

        return torch.mean(-torch.sum(y_true * preds_log, dim=1))

    
class rankNet(nn.Module):
    """
    RankNet loss introduced in "Learning to Rank using Gradient Descent".
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param weight_by_diff: flag indicating whether to weight the score differences by ground truth differences.
    :param weight_by_diff_powed: flag indicating whether to weight the score differences by the squared ground truth differences.
    :return: loss value, a torch.Tensor
    """
    def __init__(self, padded_value_indicator=PADDED_Y_VALUE, weight_by_diff=False, weight_by_diff_powed=False):
        super(rankNet, self).__init__()
        self.padded_value_indicator = padded_value_indicator
        self.weight_by_diff = weight_by_diff
        self.weight_by_diff_powed = weight_by_diff_powed

    def forward(self, y_pred, y_true):

        y_pred = y_pred.clone()
        y_true = y_true.clone()

        mask = y_true == self.padded_value_indicator
        y_pred[mask] = float('-inf')
        y_true[mask] = float('-inf')

        # here we generate every pair of indices from the range of document length in the batch
        document_pairs_candidates = list(product(range(y_true.shape[1]), repeat=2))

        pairs_true = y_true[:, document_pairs_candidates]
        selected_pred = y_pred[:, document_pairs_candidates]

        # here we calculate the relative true relevance of every candidate pair
        true_diffs = pairs_true[:, :, 0] - pairs_true[:, :, 1]
        pred_diffs = selected_pred[:, :, 0] - selected_pred[:, :, 1]

        # here we filter just the pairs that are 'positive' and did not involve a padded instance
        # we can do that since in the candidate pairs we had symetric pairs so we can stick with
        # positive ones for a simpler loss function formulation
        the_mask = (true_diffs > 0) & (~torch.isinf(true_diffs))

        pred_diffs = pred_diffs[the_mask]

        weight = None
        if self.weight_by_diff:
            abs_diff = torch.abs(true_diffs)
            weight = abs_diff[the_mask]
        elif self.weight_by_diff_powed:
            true_pow_diffs = torch.pow(pairs_true[:, :, 0], 2) - torch.pow(pairs_true[:, :, 1], 2)
            abs_diff = torch.abs(true_pow_diffs)
            weight = abs_diff[the_mask]

        # here we 'binarize' true relevancy diffs since for a pairwise loss we just need to know
        # whether one document is better than the other and not about the actual difference in
        # their relevancy levels
        true_diffs = (true_diffs > 0).type(torch.float32)
        true_diffs = true_diffs[the_mask]

        return nn.BCEWithLogitsLoss(weight=weight)(pred_diffs, true_diffs)


class lambdaLoss(nn.Module):
    """
    LambdaLoss framework for LTR losses implementations, introduced in "The LambdaLoss Framework for Ranking Metric Optimization".
    Contains implementations of different weighing schemes corresponding to e.g. LambdaRank or RankNet.
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param eps: epsilon value, used for numerical stability
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :param weighing_scheme: a string corresponding to a name of one of the weighing schemes
    :param k: rank at which the loss is truncated
    :param sigma: score difference weight used in the sigmoid function
    :param mu: optional weight used in NDCGLoss2++ weighing scheme
    :param reduction: losses reduction method, could be either a sum or a mean
    :param reduction_log: logarithm variant used prior to masking and loss reduction, either binary or natural
    :return: loss value, a torch.Tensor
    """
    def __init__(self, eps=DEFAULT_EPS, padded_value_indicator=PADDED_Y_VALUE, weighing_scheme=None, k=None, sigma=1., mu=10.,
               reduction="sum", reduction_log="binary"):
        super(lambdaLoss, self).__init__()
        self.eps = eps
        self.padded_value_indicator = padded_value_indicator
        self.weighing_scheme = weighing_scheme
        self.k = k
        self.sigma = sigma
        self.mu = mu
        self.reduction = reduction
        self.reduction_log = reduction_log

    def forward(self, y_pred, y_true):

        device = y_pred.device
        y_pred = y_pred.clone()
        y_true = y_true.clone()

        padded_mask = y_true == self.padded_value_indicator
        y_pred[padded_mask] = float("-inf")
        y_true[padded_mask] = float("-inf")

        # Here we sort the true and predicted relevancy scores.
        y_pred_sorted, indices_pred = y_pred.sort(descending=True, dim=-1)
        y_true_sorted, _ = y_true.sort(descending=True, dim=-1)

        # After sorting, we can mask out the pairs of indices (i, j) containing index of a padded element.
        true_sorted_by_preds = torch.gather(y_true, dim=1, index=indices_pred)
        true_diffs = true_sorted_by_preds[:, :, None] - true_sorted_by_preds[:, None, :]
        padded_pairs_mask = torch.isfinite(true_diffs)

        if self.weighing_scheme != "ndcgLoss1_scheme":
            padded_pairs_mask = padded_pairs_mask & (true_diffs > 0)

        ndcg_at_k_mask = torch.zeros((y_pred.shape[1], y_pred.shape[1]), dtype=torch.bool, device=device)
        ndcg_at_k_mask[:self.k, :self.k] = 1

        # Here we clamp the -infs to get correct gains and ideal DCGs (maxDCGs)
        true_sorted_by_preds.clamp_(min=0.)
        y_true_sorted.clamp_(min=0.)

        # Here we find the gains, discounts and ideal DCGs per slate.
        pos_idxs = torch.arange(1, y_pred.shape[1] + 1).to(device)
        D = torch.log2(1. + pos_idxs.float())[None, :]
        maxDCGs = torch.sum(((torch.pow(2, y_true_sorted) - 1) / D)[:, :self.k], dim=-1).clamp(min=self.eps)
        G = (torch.pow(2, true_sorted_by_preds) - 1) / maxDCGs[:, None]

        # Here we apply appropriate weighing scheme - ndcgLoss1, ndcgLoss2, ndcgLoss2++ or no weights (=1.0)
        if self.weighing_scheme is None:
            weights = 1.
        else:
            weights = globals()[weighing_scheme](G, D, mu, true_sorted_by_preds)  # type: ignore

        # We are clamping the array entries to maintain correct backprop (log(0) and division by 0)
        scores_diffs = (y_pred_sorted[:, :, None] - y_pred_sorted[:, None, :]).clamp(min=-1e8, max=1e8)
        scores_diffs.masked_fill(torch.isnan(scores_diffs), 0.)
        weighted_probas = (torch.sigmoid(self.sigma * scores_diffs).clamp(min=self.eps) ** weights).clamp(min=self.eps)
        if self.reduction_log == "natural":
            losses = torch.log(weighted_probas)
        elif self.reduction_log == "binary":
            losses = torch.log2(weighted_probas)
        else:
            raise ValueError("Reduction logarithm base can be either natural or binary")

        if self.reduction == "sum":
            loss = -torch.sum(losses[padded_pairs_mask & ndcg_at_k_mask])
        elif self.reduction == "mean":
            loss = -torch.mean(losses[padded_pairs_mask & ndcg_at_k_mask])
        else:
            raise ValueError("Reduction method can be either sum or mean")

        return loss


class approxNDCGLoss(nn.Module):
    """
    Loss based on approximate NDCG introduced in "A General Approximation Framework for Direct Optimization of
    Information Retrieval Measures". Please note that this method does not implement any kind of truncation.
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param eps: epsilon value, used for numerical stability
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :param alpha: score difference weight used in the sigmoid function
    :return: loss value, a torch.Tensor
    """
    def __init__(self, eps=DEFAULT_EPS, padded_value_indicator=PADDED_Y_VALUE, alpha=1.):
        super(approxNDCGLoss, self).__init__()
        self.eps = eps
        self.padded_value_indicator = padded_value_indicator
        self.alpha = alpha

    def forward(self, y_pred, y_true):

        device = y_pred.device
        y_pred = y_pred.clone()
        y_true = y_true.clone()

        padded_mask = y_true == self.padded_value_indicator
        y_pred[padded_mask] = float("-inf")
        y_true[padded_mask] = float("-inf")

        # Here we sort the true and predicted relevancy scores.
        y_pred_sorted, indices_pred = y_pred.sort(descending=True, dim=-1)
        y_true_sorted, _ = y_true.sort(descending=True, dim=-1)

        # After sorting, we can mask out the pairs of indices (i, j) containing index of a padded element.
        true_sorted_by_preds = torch.gather(y_true, dim=1, index=indices_pred)
        true_diffs = true_sorted_by_preds[:, :, None] - true_sorted_by_preds[:, None, :]
        padded_pairs_mask = torch.isfinite(true_diffs)
        padded_pairs_mask.diagonal(dim1=-2, dim2=-1).zero_()

        # Here we clamp the -infs to get correct gains and ideal DCGs (maxDCGs)
        true_sorted_by_preds.clamp_(min=0.)
        y_true_sorted.clamp_(min=0.)

        # Here we find the gains, discounts and ideal DCGs per slate.
        pos_idxs = torch.arange(1, y_pred.shape[1] + 1).to(device)
        D = torch.log2(1. + pos_idxs.float())[None, :]
        maxDCGs = torch.sum((torch.pow(2, y_true_sorted) - 1) / D, dim=-1).clamp(min=self.eps)
        G = (torch.pow(2, true_sorted_by_preds) - 1) / maxDCGs[:, None]

        # Here we approximate the ranking positions according to Eqs 19-20 and later approximate NDCG (Eq 21)
        scores_diffs = (y_pred_sorted[:, :, None] - y_pred_sorted[:, None, :])
        scores_diffs[~padded_pairs_mask] = 0.
        approx_pos = 1. + torch.sum(padded_pairs_mask.float() * (torch.sigmoid(-self.alpha * scores_diffs).clamp(min=self.eps)), dim=-1)
        approx_D = torch.log2(1. + approx_pos)
        approx_NDCG = torch.sum((G / approx_D), dim=-1)

        return -torch.mean(approx_NDCG)


class Closs(nn.Module):
    def __init__(self):
        super(Closs, self).__init__()
    def forward(self, f, num_stocks):
        l = torch.sum(f[:,num_stocks // 2:], dim = 1) - torch.sum(f[:, :num_stocks // 2], dim = 1)
        for i in range(num_stocks // 2):
            l += torch.logsumexp(f[:,i:num_stocks-i], dim = 1)
            l += torch.logsumexp(torch.neg(f[:,i:num_stocks-i]), dim = 1)
        l = torch.mean(l)
        return l


class Closs_explained(nn.Module):
    def __init__(self):
        super(Closs_explained, self).__init__()
    def forward(self, f, num_stocks):
        l = torch.sum(f[:,num_stocks // 2:], dim = 1) - torch.sum(f[:, :num_stocks // 2], dim = 1)
        for i in range(num_stocks // 2):
            subtract = torch.tensor(num_stocks - 2*i,requires_grad = False)
            l += torch.log(torch.sum(torch.exp(f[:,i:num_stocks-i]), dim = 1)*torch.sum(torch.exp(torch.neg(f[:,i:num_stocks-i])), dim = 1)-subtract)
        l = torch.mean(l)
        return l


class Closs_sigmoid(nn.Module):
    def __init__(self):
        super(Closs_sigmoid, self).__init__()
    def forward(self, f, num_stocks):
        l = torch.tensor(1, requires_grad=False)+torch.exp(f[:,num_stocks//2:] - f[:,:num_stocks//2])
        return torch.mean(torch.log(l))


class Lloss(nn.Module):
    def __init__(self):
        super(Lloss, self).__init__()
    def forward(self, f, num_stocks):
        l = torch.neg(torch.sum(f, dim = 1))
        for i in range(num_stocks):
            l += torch.logsumexp(f[:,i:], dim = 1)
        l = torch.mean(l)
        return l


class TRR(nn.Module):
    """
    Temporal Relational Ranking
    """
    def __init__(self, alpha: float=1.0):
        super(TRR, self).__init__()
        self.alpha = alpha
        
    def forward(self, pred_y, true_y):
        reg_loss = nn.MSELoss()(pred_y, true_y)
        all_one = torch.ones((pred_y.size(0), 1))
        pre_pw_dif = pred_y.matmul(all_one.permute(1,0)) - all_one.matmul(pred_y.permute(1,0))
        gt_pw_dif = true_y.matmul(all_one.permute(1,0)) - all_one.matmul(true_y.permute(1,0))
        rank_loss = (F.relu(pre_pw_dif * gt_pw_dif)).mean()
        loss = reg_loss + self.alpha * rank_loss
        return loss

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
def read_return_data(stocklist_dir: str, save_dir: str, start_date: int, end_date: int, T: int):

    # create the folder if exists
    if not os.path.exists(f'{save_dir}/Data/Return/{T}'):
        os.makedirs(f'{save_dir}/Data/Return/{T}')

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
            stockDF['return'] = stockDF['close'].shift(-T) / stockDF['close'] - 1
            stockDF = stockDF.loc[(stockDF['date'] >= start_date) & (stockDF['date'] <= end_date)]
            stockDF = stockDF.loc[:,['id', 'date', 'return']]
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
            stock_i.to_csv(f'{save_dir}/Data/Return/{T}/{date_i}.csv', index=False)

# read the stock data
def read_price_data(stocklist_dir: str, save_dir: str, start_date: int, end_date: int):

    # create the folder if exists
    if not os.path.exists(f'{save_dir}/Data/Price'):
        os.makedirs(f'{save_dir}/Data/Price')

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
            date_bar.set_description(f'Saving price data on trading date {date_i}')
            stock_i = StockData.loc[StockData['date'] == date_i]
            stock_i.to_csv(f'{save_dir}/Data/Price/{date_i}.csv', index=False)

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

def return_rank(a):
    a = a * -1
    order = a.argsort()
    return order.argsort()

def random_batch(x, y):
	ind = np.random.randint(0, len(x), BATCH_SIZE)
	batch_x, batch_y = x[ind], y[ind]
	x_sorted = np.zeros(batch_x.shape)
	for i in range(len(batch_x)):
		rank_temp = return_rank(batch_y[i])
		rank2ind = np.zeros(N_STOCKS, dtype = int)
		for j in range(len(rank_temp)):
			rank2ind[rank_temp[j]] = int(j)
		for j in range(len(rank_temp)):
			x_sorted[i,rank_temp[j],:] = batch_x[i][rank2ind[rank_temp[j]]]
	return x_sorted

def random_batch_mlp(x, y):
	ind = np.random.randint(0, len(x), BATCH_SIZE)
	batch_x, batch_y = x[ind].astype(np.float), y[ind].astype(np.float)
	return batch_x, batch_y

# save the feature matrix data for further use
def get_rolling_feature_data(save_dir: str, feature_dir: str, trDays_dir: str, return_dir: str, member_dir: str, modelLst: list, loss: str, 
                start_year: int, end_year: int, train_window: int, test_window: int, bench: str, name: str, n_stock: int):

    # create folder to store the factor data
    if not os.path.exists(f'{save_dir}/Feature/{name}/{bench}'):
        os.makedirs(f'{save_dir}/Feature/{name}/{bench}')

    if not os.path.exists(f'{save_dir}/Model/{bench}'):
        os.makedirs(f'{save_dir}/Model/{bench}')

    # get the trading date sequence
    start_date, end_date = int(f'{start_year}0101'), int(f'{end_year}1231')
    trDate = read_pkl5_data(trDays_dir)
    trDate = trDate.index.to_numpy()
    trDate = trDate[(trDate >= start_date) & (trDate <= end_date)]

    # return_df = pd.DataFrame()
    # # load the training dataset
    # with trange(len(trDate)) as date_bar:
    #     for i in date_bar:
    #         date_i = trDate[i]
    #         date_bar.set_description(f'Loading data on date {date_i}')

    #         # get data from different datasets
    #         return_i = pd.read_csv(f'{return_dir}/{date_i}.csv')
    #         return_df = return_df.append(return_i)

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

        if os.path.exists(f'{save_dir}/Feature/{name}/{bench}/{train_dates[0]}_{train_dates[-1]}_data.npy'):
            train_feature, train_id, train_date, train_return = np.load(f'{save_dir}/Feature/{name}/{bench}/{train_dates[0]}_{train_dates[-1]}_data.npy', allow_pickle=True)
        else:
            train_feature, train_id, train_date, train_return = load_dataset(feature_dir, train_dates, return_df, name, member_df, n_stock)
            # train_feature, a = get_preprocess(train_feature)
            train_data = np.empty(4, dtype=object)
            train_data[:] = [train_feature, train_id, train_date, train_return]
            # save the feature data for further use
            np.save(f'{save_dir}/Feature/{name}/{bench}/{train_dates[0]}_{train_dates[-1]}_data.npy', train_data, allow_pickle=True)

        if os.path.exists(f'{save_dir}/Feature/{name}/{bench}/{test_dates[0]}_{test_dates[-1]}_data.npy'):
            test_feature, test_id, test_date, test_return = np.load(f'{save_dir}/Feature/{name}/{bench}/{test_dates[0]}_{test_dates[-1]}_data.npy', allow_pickle=True)
        else:
            test_feature, test_id, test_date, test_return = load_dataset(feature_dir, test_dates, return_df, name, member_df, n_stock)
            # test_feature = preprocess(test_feature, a)
            test_data = np.empty(4, dtype=object)
            test_data[:] = [test_feature, test_id, test_date, test_return]
            # save the feature data for further use
            np.save(f'{save_dir}/Feature/{name}/{bench}/{test_dates[0]}_{test_dates[-1]}_data.npy', test_data, allow_pickle=True)

        train_feature, test_feature = np.nan_to_num(train_feature, nan=0), np.nan_to_num(test_feature, nan=0)

        train_return = pd.DataFrame({'id': train_id, 'date': train_date, 'return': train_return})
        train_return.set_index(['id', 'date'], inplace=True)
        train_return = train_return.unstack().values
        
        train_return = np.nan_to_num(train_return, nan=0)
        train_return = np.transpose(train_return, (1, 0))
        n_feature = train_feature.shape[-1]
        train_return += 1
        train_return = np.log(train_return)

        mask = ~(train_feature == 0).all(axis=(1, 2))
        train_feature = train_feature[mask]
        train_return = train_return[mask]

        # train the model one by one
        for model_name in modelLst:
            if not os.path.exists(f'{save_dir}/Model/{bench}/{model_name}_{loss}_{year_i}_{year_i+train_window-1}.dat'):
                # choose the right model to train
                if model_name == 'DNN':
                    model = CMLE(n_features=n_feature)
                elif model_name == 'LSTM':
                    model = LSTMModel(input_size=n_feature)
                elif model_name == 'GRU':
                    model = GRUModel(input_size=n_feature)
                elif model_name == 'ALSTM':
                    model = ALSTMModel(input_size=n_feature)
                elif model_name == 'Transformer':
                    model = make_transformer(n_features=n_feature)
                elif model_name == 'LTR':
                    model = make_ltr_model(fc_model=True, transformer=True, n_features=n_feature)
                else:
                    raise ValueError(f'The parameter model should be LSTM/GRU/ALSTM/TCN/Transformer, get {model_name} instead')

                model.to(device=device)
                model.train()
                model = model.double()

                # choose the right loss function to use
                if loss == 'Closs':
                    loss_func = Closs()
                    train(model, loss_func, train_feature, train_return, n_stock, year_i, train_window, bench, save_dir, model_name, loss)
                elif loss == 'Closs_explained':
                    loss_func = Closs_explained()
                    train(model, loss_func, train_feature, train_return, n_stock, year_i, train_window, bench, save_dir, model_name, loss)
                elif loss == 'Closs_sigmoid':
                    loss_func = Closs_sigmoid()
                    train(model, loss_func, train_feature, train_return, n_stock, year_i, train_window, bench, save_dir, model_name, loss)
                elif loss == 'Lloss':
                    loss_func = Lloss()
                    train(model, loss_func, train_feature, train_return, n_stock, year_i, train_window, bench, save_dir, model_name, loss)
                elif loss == 'MSE':
                    loss_func = nn.MSELoss()
                    train_mlp(model, loss_func, train_feature, train_return, year_i, train_window, bench, save_dir, model_name, loss)
                elif loss == 'ListMLE':
                    loss_func = ListMLE()
                    train_mlp(model, loss_func, train_feature, train_return, year_i, train_window, bench, save_dir, model_name, loss)
                elif loss == 'ListNet':
                    loss_func = ListNet()
                    train_mlp(model, loss_func, train_feature, train_return, year_i, train_window, bench, save_dir, model_name, loss)
                elif loss == 'RankNet':
                    loss_func = rankNet()
                    train_mlp(model, loss_func, train_feature, train_return, year_i, train_window, bench, save_dir, model_name, loss)
                elif loss == 'LambdaLoss':
                    loss_func = lambdaLoss()
                    train_mlp(model, loss_func, train_feature, train_return, year_i, train_window, bench, save_dir, model_name, loss)
                elif loss == 'ApproxNDCG':
                    loss_func = approxNDCGLoss()
                    train_mlp(model, loss_func, train_feature, train_return, year_i, train_window, bench, save_dir, model_name, loss)
                elif loss == 'BinaryListNet':
                    loss_func = binary_listNet()
                    train_mlp(model, loss_func, train_feature, train_return, year_i, train_window, bench, save_dir, model_name, loss)
                elif loss == 'TRR':
                    loss_func = TRR()
                    train_mlp(model, loss_func, train_feature, train_return, year_i, train_window, bench, save_dir, model_name, loss)
                else:
                    raise ValueError(f'The parameter loss should be Closs/Closs Explained/Closs Sigmoid/Lloss, get {loss} instead')


def train(model, loss_func, train_feature, train_return, n_stock, year_i, train_window, bench, save_dir, model_name, loss):
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    running_loss = []
    torch.set_grad_enabled(True)

    # training data by epoch
    with trange(MAX_EPOCH) as train_bar:
        for i in train_bar:
            train_bar.set_description(f'Training epoch {i+1}')
            batch_x = Variable(torch.from_numpy(random_batch(train_feature, train_return)).double())
            batch_x = batch_x.to(device=device)
            model.train()
            scores = model(batch_x)
            # calculate the loss
            l = loss_func(scores, torch.tensor(n_stock, requires_grad = False))
            # clear the gradients
            optimizer.zero_grad()
            # credit assignment
            l.backward()
            # update model weights
            optimizer.step()
            # store the running loss
            running_loss.append(float(l))
            # save the model state after training some epochs
            if (i+1) % EPOCH == 0:
                train_bar.set_postfix(train_loss=np.mean(running_loss))
                running_loss = []
                torch.save(model.state_dict(), f'{save_dir}/Model/{bench}/{model_name}_{loss}_{year_i}_{year_i+train_window-1}.dat')


def train_mlp(model, loss_func, train_feature, train_return, year_i, train_window, bench, save_dir, model_name, loss):
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    running_loss = []
    torch.set_grad_enabled(True)

    # training data by epoch
    with trange(MAX_EPOCH) as train_bar:
        for i in train_bar:
            train_bar.set_description(f'Training epoch {i+1}')
            batch_x, batch_y = random_batch_mlp(train_feature, train_return)
            batch_x, batch_y = Variable(torch.from_numpy(batch_x)), Variable(torch.from_numpy(batch_y))
            batch_x, batch_y = batch_x.to(device=device), batch_y.to(device=device)
            model.train()
            scores = model(batch_x)
            # calculate the loss
            l = loss_func(scores, batch_y)
            # clear the gradients
            optimizer.zero_grad()
            # credit assignment
            l.backward()
            # update model weights
            optimizer.step()
            # store the running loss
            running_loss.append(float(l))
            # save the model state after training some epochs
            if (i+1) % EPOCH == 0:
                train_bar.set_postfix(train_loss=np.mean(running_loss))
                running_loss = []
                torch.save(model.state_dict(), f'{save_dir}/Model/{bench}/{model_name}_{loss}_{year_i}_{year_i+train_window-1}.dat')


def test(model, test_features):

    L = len(test_features)
    N = len(test_features) // BATCH_SIZE + 1
    v = np.zeros((N*BATCH_SIZE, test_features.shape[1], test_features.shape[2]))
    v[:L, :, :] = test_features
    for i in range(N * BATCH_SIZE - L):
        v[i+L,:,:] = test_features[0,:,:]
    res = []
    for i in range(N):
        batch_x = Variable(torch.from_numpy(v[i * BATCH_SIZE:(i+1) * BATCH_SIZE,:,:]).double())
        scores = model(batch_x)
        res.append(np.array(scores.data.cpu()))
    res = np.concatenate(res, axis = 0)
    res = res[:L]

    return res


def back_test(k, score, returns, n_stock, short='bottom'):

    res, bench_res = [], []
    weight_list_pos, weight_list_neg = [], []
    return_list_pos, return_list_neg = [], []
    for i in range(len(score)):
        rank = return_rank(score[i])
        rank2ind = np.zeros(len(rank), dtype = int)
        for j in range(len(rank)):
            rank2ind[rank[j]] = j
        weights = np.zeros(k)
        for j in range(k):
            weights[j] = score[i][rank2ind[j]]
        s = k * (k+1) / 2.0
        for j in range(k):
            weights[j] = (k - j) / s
            weights[j] = 1.0 / k
        total_return = 0
        for j in range(k):
            total_return += weights[j] * returns[i][rank2ind[j]]
            if short == 'bottom':
                total_return -= weights[j] * returns[i][rank2ind[n_stock - 1 - j]]
        if short == 'average':
            for j in range(n_stock):
                total_return -= 1.0 / n_stock * returns[i][rank2ind[j]]
        res.append(total_return)
        bench_res.append(returns[i].mean())
        pos, neg = [], []
        r_pos, r_neg = [], []
        for j in range(k):
            pos.append(rank2ind[j])
            neg.append(rank2ind[n_stock - 1 - j])
            r_pos.append(returns[i][rank2ind[j]])
            r_neg.append(returns[i][rank2ind[n_stock - 1 - j]])
        weight_list_pos.append(pos)
        weight_list_neg.append(neg)
        return_list_pos.append(r_pos)
        return_list_neg.append(r_neg)

    return np.array(res), np.array(bench_res), np.array(weight_list_pos), np.array(weight_list_neg), np.array(return_list_pos), np.array(return_list_neg)


def load_model_test_ranks(model, model_name, test_features, test_ranks, ranks, short='bottom'):

    saved_state = torch.load(model_name)
    model.load_state_dict(saved_state)
    n_stock = test_features.shape[0]
    y_pred = test(model, test_features)
    y = np.array(y_pred)
    r = test_ranks
    r = r[:len(y), :]
    res, bench_res, weight_list_pos, weight_list_neg, return_list_pos, return_list_neg = back_test(ranks,y,r,n_stock,short)
    
    return res, bench_res, weight_list_pos, weight_list_neg, return_list_pos, return_list_neg


def get_predict(save_dir: str, trDays_dir: str, modelLst: list, loss: str, start_date: int, end_date: int, short: str, N: int, rank_to_use: int, T: int,
                start_year: int, end_year: int, train_window: int, test_window: int, bench: str, name: str):
    
    # create folder to store the factor data
    if not os.path.exists(f'{save_dir}/Feature/{name}/{bench}'):
        os.makedirs(f'{save_dir}/Feature/{name}/{bench}')

    if not os.path.exists(f'{save_dir}/Model/{bench}'):
        os.makedirs(f'{save_dir}/Model/{bench}')

    # get the trading date sequence
    trDate = read_pkl5_data(trDays_dir)
    trDate = trDate.index.to_numpy()
    trDate = trDate[(trDate >= start_date) & (trDate <= end_date)]

    test_feature, test_date, test_return = [], [], []
    # iterately train and test the model
    for year_i in range(start_year, end_year-train_window-test_window+2):

        print(f'''Loading data with training period {year_i}-{year_i+train_window-1}
        and testing period {year_i+train_window}-{year_i+train_window+test_window-1}''')

        # get the dates for corresponding data sets
        test_dates = trDate[(int(f'{year_i+train_window}0101') <= trDate) 
                            & (trDate <= int(f'{year_i+train_window+test_window-1}1231'))]
        
        test_feature_i, test_id_i, test_date_i, test_return_i = np.load(f'{save_dir}/Feature/{name}/{bench}/{test_dates[0]}_{test_dates[-1]}_data.npy', 
                                                                allow_pickle=True)
        
        n_feature = test_feature_i.shape[-1]
        # test_return = test_return.values
        # test_return += 1
        # test_return = np.log(test_return)
        test_return_i = pd.DataFrame({'id': test_id_i, 'date': test_date_i, 'return': test_return_i})
        test_return_i.set_index(['id', 'date'], inplace=True)
        test_return_i = test_return_i.unstack()
        test_Y = test_return_i.values
        
        test_Y = np.nan_to_num(test_Y, nan=0)
        test_Y = np.transpose(test_Y, (1, 0))

        dates = [test_dates[i] for i in range(0,len(test_dates),T)]
        dates_col = [('return', d) for d in dates]
        col_idx = test_return_i.columns.isin(dates_col)
        dates = np.array(dates)

        test_feature_i = test_feature_i[col_idx,:,:]
        test_Y = test_Y[col_idx,:]

        test_feature.append(test_feature_i)
        test_date.append(dates)
        test_return.append(test_Y)

    test_feature = np.concatenate(test_feature, axis=0)
    test_date = np.concatenate(test_date, axis=0)
    test_return = np.concatenate(test_return, axis=0)
        
    # train the model one by one
    for model_name in modelLst:
        # choose the right model to train
        if model_name == 'DNN':
            model = CMLE(n_features=n_feature)
        elif model_name == 'LSTM':
            model = LSTMModel(input_size=n_feature, output_size=1)
        elif model_name == 'GRU':
            model = GRUModel(input_size=n_feature, output_size=1)
        elif model_name == 'ALSTM':
            model = ALSTMModel(input_size=n_feature, output_size=1)
        else:
            raise ValueError(f'The parameter model should be LSTM/GRU/ALSTM/TCN/Transformer, get {model_name} instead')

        model.eval()
        model = model.double()
        
        d = pd.DataFrame()
        tt, bench_tt = [], []
        wp, wn, rp, rn = [], [], [], []

        for _ in range(0, N):
            model_states_dir = f'{save_dir}/Model/{bench}/{model_name}_{loss}_{year_i}_{year_i+train_window-1}.dat'
            tmp, bench_tmp, weight_list_pos, weight_list_neg, return_list_pos, return_list_neg = load_model_test_ranks(model, model_states_dir, test_feature, test_return, rank_to_use, short)
            tt.append(tmp)
            bench_tt.append(bench_tmp)
            wp.append(weight_list_pos)
            wn.append(weight_list_neg)
            rp.append(return_list_pos)
            rn.append(return_list_neg)

        if not os.path.exists(f'{save_dir}/results'):
            os.makedirs(f'{save_dir}/results')

        tt = np.concatenate(tt)
        bench_tt = np.concatenate(bench_tt)
        wp = np.concatenate(wp, axis=0)
        wn = np.concatenate(wn, axis=0)
        rp = np.concatenate(rp, axis=0)
        rn = np.concatenate(rn, axis=0)

        d['date'] = test_date

        for i in range(rank_to_use):
            d['pos_ticker_'+str(i+1)] = wp[:,i]
        for i in range(rank_to_use):
            d['neg_ticker_'+str(i+1)] = wn[:,i]

        d['return'] = tt
        d['benchmark'] = bench_tt
        d.to_csv(f'{save_dir}/results/{model_name}_{loss}_{short}_{rank_to_use}_{test_date[0]}_{test_date[-1]}.csv', index = False)


def qs_backtest_bench(save_dir: str, model_name: str, loss: str, short: str, rank_to_use: int,
                start_date: int, end_date: int, bench: str, name: str):
    
    # create folder to store the factor data
    if not os.path.exists(f'{save_dir}/Report/{name}/{bench}'):
        os.makedirs(f'{save_dir}/Report/{name}/{bench}')
    
    # bench_df[bench_col] /= 100

    df = pd.read_csv(f'{save_dir}/results/{model_name}_{loss}_{short}_{rank_to_use}_{start_date}_{end_date}.csv')
    df = df[['date', 'return', 'benchmark']]

    # df['return'] += 1
    # print(df['return'].prod())
    start_date, end_date = df['date'].iloc[0], df['date'].iloc[-1]
    df['date'] = df['date'].apply(lambda date_i: pd.Timestamp(int(str(date_i)[:4]), int(str(date_i)[4:6]), int(str(date_i)[6:])))
    df.set_index('date', inplace=True)

    print(df)

    report_dir = f'{save_dir}/Report/{name}/{bench}/{model_name}_{loss}_{short}_{rank_to_use}_{start_date}_{end_date}.html'

    qs.reports.html(df['return'], df['benchmark'],
        title=f'Report of long-short portfolio with factor predicted by {model_name}',
        output=report_dir)

    print('Report saved in %s' % (report_dir))