import os
import sys
import pickle5
import numpy as np
import pandas as pd
import quantstats as qs
import joblib
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from functools import reduce
from tqdm import trange
from VGG import *
from ResNet import *

sys.path.append('E:/实习/南方202209/CNN股票图片/vit_pytorch')

import vit
import vit_1d
import cct
import cvt
import deepvit
import learnable_memory_vit
import levit
import local_vit
import max_vit
import mobile_vit
import parallel_vit
import pit
import regionvit
import rvt
import sep_vit
import simple_flash_attn_vit
import simple_vit
import simple_vit_1d
import simple_vit_with_patch_dropout
import twins_svt
import vit_for_small_dataset


IMAGE_WIDTH = {5: 15, 20: 60, 60: 180}
IMAGE_HEIGHT = {5: 32, 20: 64, 60: 96}        

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Variables
# BATCH_SIZE = 1024
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-5
MAX_EPOCH = 5
# MAX_EPOCH = 30
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


class CNN20d(nn.Module):
    def __init__(self, output_size=2):
        super(CNN20d, self).__init__()

        self.output_size = output_size
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(5,3), stride=(3,1), dilation=(2,1))
        self.pool1 = nn.MaxPool2d(kernel_size=(2,1), stride=(1,1))
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5,3))
        self.pool2 = nn.MaxPool2d(kernel_size=(2,1), stride=(1,1))
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(5,3))
        self.pool3 = nn.MaxPool2d(kernel_size=(2,1), stride=(1,1))
        self.dropout = nn.Dropout(p=0.25)
        self.fc = nn.Linear(256 * 54 * 8, output_size)
        self.init_weights()
        self.net1 = nn.Sequential(self.conv1, nn.BatchNorm2d(64), nn.LeakyReLU(), self.pool1)
        self.net2 = nn.Sequential(self.conv2, nn.BatchNorm2d(128), nn.LeakyReLU(), self.pool2)
        self.net3 = nn.Sequential(self.conv3, nn.BatchNorm2d(256), nn.LeakyReLU(), self.pool3)

    def init_weights(self):
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        x = x.unsqueeze(dim=1) # (B, 1, 64, 60)
        x = self.net1(x) # (B, 64, 19, 58), (B, 64, 18, 58)
        x = self.net2(x) # (B, 128, 14, 56), (B, 128, 13, 56) 
        x = self.net3(x) # (B, 256, 9, 54), (B, 256, 8, 54)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        if self.output_size != 1:
            x = F.softmax(x, dim=1)
        return x

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


def get_MA_data(save_dir: str, close_dir: str, start_date: int):

    close_df = stack_pkl_data(close_dir, start_date, 'close')
    close_df.sort_values(['id', 'date'], inplace=True)
    close_df.reset_index(drop=True, inplace=True)
    close_df['MA'] = close_df.groupby('id')['close'].rolling(20).mean().reset_index(drop=True)
    del close_df['close']
    close_df.dropna(subset=['MA'], inplace=True)
    dateLst = np.sort(pd.unique(close_df['date']))
    n_stock = len(pd.unique(close_df['id']))
    start, end = dateLst[0], dateLst[-1]
    close_df.set_index(['id', 'date'], inplace=True)
    close_df = close_df.unstack()
    write_pkl5_data(f'{save_dir}/MA_WideMat_{n_stock}_{start}_{end}.pkl', close_df)

# transform the data in matrix form into three columns dataframe
def stack_pkl_data(dir: str, start_date: int, feature: str):

    df = read_pkl5_data(dir)
    df = df.stack().reset_index()
    df.rename(columns={'level_0': 'date', 'level_1': 'id', 0: feature}, inplace=True)
    df = df.loc[df['date'] >= start_date]
    df['id'] = df['id'].apply(lambda x: x[:-3])
    df = df.loc[df['id'].apply(lambda x: x.isdigit())]
    df['id'] = df['id'].astype(np.int32)
    return df


def get_20d_data(save_dir: str, trDays_dir: str, return_dir: str, start_year: int, end_year: int, 
                 open_dir: str, close_dir: str, high_dir: str, low_dir: str, volume_dir: str, MA_dir: str):

    # create folder to store the factor data
    if not os.path.exists(f'{save_dir}/20Day'):
        os.makedirs(f'{save_dir}/20Day')

    # get the trading date sequence
    trDate = read_pkl5_data(trDays_dir)
    trDate = trDate.index.to_numpy()
    trDate = trDate[(trDate >= int(f'{start_year}0101')) & (trDate <= int(f'{end_year}1231'))]

    return_df = pd.DataFrame()
    # load the training dataset
    with trange(len(trDate)) as date_bar:
        for i in date_bar:
            date_i = trDate[i]
            date_bar.set_description(f'Loading data on date {date_i}')

            # get data from different datasets
            return_i = pd.read_csv(f'{return_dir}/{date_i}.csv')
            return_df = return_df.append(return_i)

    start_date = int(f'{start_year}0101')
    open_df = stack_pkl_data(open_dir, start_date, 'open')
    close_df = stack_pkl_data(close_dir, start_date, 'close')
    high_df = stack_pkl_data(high_dir, start_date, 'high')
    low_df = stack_pkl_data(low_dir, start_date, 'low')
    volume_df = stack_pkl_data(volume_dir, start_date, 'volume')

    MA_df = read_pkl5_data(MA_dir)
    MA_df = MA_df.stack().reset_index()
    MA_df.rename(columns={'level_0': 'date', 'level_1': 'id', 0: 'MA'}, inplace=True)
    MA_df = MA_df.loc[MA_df['date'] >= start_date]

    for y in range(start_year, end_year+1):

        start_i, end_i = int(f'{y}0101'), int(f'{y}1231')
        trDate_i = trDate[(trDate >= start_i) & (trDate <= end_i)]

        dataLst, ret_df = [], pd.DataFrame()
        with trange(0, len(trDate_i)-20, 20) as date_bar:
            for i in date_bar:
                start_ii, end_ii = trDate_i[i], trDate_i[i+20]
                date_bar.set_description(f'Process data between {start_ii} and {end_ii}')

                open_i = open_df.loc[(open_df['date'] >= start_ii) & (open_df['date'] < end_ii)].copy()
                close_i = close_df.loc[(close_df['date'] >= start_ii) & (close_df['date'] < end_ii)].copy()
                high_i = high_df.loc[(high_df['date'] >= start_ii) & (high_df['date'] < end_ii)].copy()
                low_i = low_df.loc[(low_df['date'] >= start_ii) & (low_df['date'] < end_ii)].copy()
                volume_i = volume_df.loc[(volume_df['date'] >= start_ii) & (volume_df['date'] < end_ii)].copy()
                MA_i = MA_df.loc[(MA_df['date'] >= start_ii) & (MA_df['date'] < end_ii)].copy()

                df_i = reduce(lambda df1, df2: pd.merge(df1, df2, on=['id', 'date'], how='inner'), [open_i, close_i, high_i, low_i, volume_i, MA_i])
                df_i.sort_values(['id', 'date'], inplace=True)
                date_i = df_i['date'].iloc[-1]
                idLst_i = np.sort(pd.unique(df_i['id']))
                for id_i in idLst_i:
                    data_i = np.zeros((64, 60))
                    ret_i = return_df.loc[(return_df['date'] == date_i) & (return_df['id'] == id_i)]
                    df_ii = df_i.loc[df_i['id'] == id_i].copy()
                    if df_ii[['open', 'high', 'low', 'close', 'MA']].isnull().all(axis=0).any():
                        continue
                    if df_ii.shape[0] != 20:
                        continue
                    if ret_i.empty:
                        continue

                    max_i, min_i = df_ii[['high', 'MA']].max().max(), df_ii[['low', 'MA']].min().min()

                    df_ii[['open', 'high', 'low', 'close', 'MA']] = (df_ii[['open', 'high', 'low', 'close', 'MA']] - min_i) / (max_i - min_i) * 50 + 13
                    df_ii[['open', 'high', 'low', 'close', 'MA']] = df_ii[['open', 'high', 'low', 'close', 'MA']].fillna(65)
                    df_ii[['open', 'high', 'low', 'close', 'MA']] = df_ii[['open', 'high', 'low', 'close', 'MA']].astype(np.int8)
                    open_ii = df_ii['open'].values
                    close_ii = df_ii['close'].values
                    high_ii = df_ii['high'].values
                    low_ii = df_ii['low'].values

                    max_vol_i = df_ii['volume'].max()
                    df_ii['volume'] = df_ii['volume'] / max_vol_i * 12
                    df_ii['volume'] = df_ii['volume'].fillna(65)
                    df_ii['volume'] = df_ii['volume'].astype(np.int8)
                    volume_ii = df_ii['volume'].values

                    ma_i = df_ii['MA'].values
                    ma_ids = [3*i+1 for i in range(20)]
                    not_ma_ids = [i for i in range(60) if i % 3 != 1]
                    not_ma_i = np.interp(not_ma_ids, ma_ids, ma_i)
                    ma_i = np.concatenate([ma_i, not_ma_i])
                    ma_ids = ma_ids + not_ma_ids
                    ma_i = ma_i[np.argsort(ma_ids)]
                    ma_i = np.int8(ma_i)

                    for j in range(20):
                        try:
                            data_i[j*3][open_ii[j]] = 255
                        except:
                            pass
                        try:
                            data_i[j*3+1][low_ii[j]:high_ii[j]] = 255
                        except:
                            pass
                        try:
                            data_i[j*3+2][close_ii[j]] = 255
                        except:
                            pass
                        try:
                            data_i[j*3+1][:volume_ii[j]] = 255
                        except:
                            pass
                        try:
                            data_i[j*3][ma_i[j*3]] = 255
                        except:
                            pass
                        try:
                            data_i[j*3+1][ma_i[j*3+1]] = 255
                        except:
                            pass
                        try:
                            data_i[j*3+2][ma_i[j*3+2]] = 255
                        except:
                            pass

                    data_i = np.flipud(data_i.T)

                    # plt.imshow(data_i, cmap='gray')
                    # plt.show()
                    dataLst.append(data_i)
                    ret_df = ret_df.append(ret_i)

        data = np.stack(dataLst, axis=0)
        print(data.shape)
        print(ret_df.shape)
        write_pkl5_data(f'{save_dir}/20Day/{y}_image.pkl', data)
        write_pkl5_data(f'{save_dir}/20Day/{y}_label.pkl', ret_df)


class BCE(nn.Module):
    def __init__(self):
        super(BCE, self).__init__()
    def forward(self, y_pred, y_true):
        return F.binary_cross_entropy(y_pred[:,-1], y_true)
    
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
    
# CCC loss function, a combination of MSE loss and IC loss
class CCC_loss(nn.Module):
    def __init__(self):
        super(CCC_loss, self).__init__()

    def forward(self, logits, target):
        logits_mean, target_mean = logits.mean(), target.mean()
        logits_var, target_var = ((logits - logits_mean) ** 2).sum(), ((target - target_mean) ** 2).sum()
        denomi = torch.sum((logits - logits_mean) * (target - target_mean))
        ccc = 2 * denomi / (logits_var + target_var + (logits_mean - target_mean) ** 2)
        return -ccc
    
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
            x_test, y_test = x_test.float(), y_test.float()

            # compute the model output without calculating the gradient
            with torch.no_grad():
                y_pred = model(x_test)
                y_pred, y_test = y_pred.double(), y_test.double()

            # calculate loss
            Loss += criterion(y_pred, y_test).item()

            # set information for the bar
            test_bar.set_postfix(evaluate_loss=Loss / (i+1))

            # delete data to release memory
            del x_test, y_test
            torch.cuda.empty_cache()

        return Loss / (i+1)
    
# train the nn model
def train(model, train_dataloader, criterion,  valid_dataloader=None, MAX_EPOCH=MAX_EPOCH):

    Best_model = None
    min_loss = np.inf

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
                x_train, y_train = x_train.float(), y_train.float()

                # clear the gradients
                optimizer.zero_grad()

                # compute the model output
                y_pred = model(x_train)
                y_pred, y_train = y_pred.double(), y_train.double()

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
                model.train()
                if loss < min_loss:
                    Best_model = model
                    min_loss = loss
            else:
                Best_model = model

    return Best_model, Total_loss

# return the retun of array
def return_rank(a):
    a = a * -1
    order = a.argsort()
    return order.argsort()

# train the nn model
def train_rank(model, train_dataloader, valid_dataloader, criterion, n_stocks, MAX_EPOCH=MAX_EPOCH):

    Best_model = None
    min_loss = np.inf

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
                x_train, y_train = x_train.float(), y_train.float()
                # sort features based on label
                x_sorted = torch.zeros(x_train.shape)
                for i in range(x_train.size(0)):
                    rank_temp = return_rank(y_train[i])
                    rank2ind = torch.zeros(n_stocks, dtype = int)
                    for j in range(rank_temp.size(0)):
                        rank2ind[rank_temp[j]] = int(j)
                    for j in range(rank_temp.size(0)):
                        x_sorted[i,rank_temp[j],:] = x_train[i][rank2ind[rank_temp[j]]]
                x_sorted = x_sorted.to(device)
                y_train = torch.tensor(n_stocks, requires_grad = False)

                # clear the gradients
                optimizer.zero_grad()

                # compute the model output
                y_pred = model(x_train)
    
                y_pred, y_train = y_pred.double(), y_train.double()

                # calculate loss
                train_loss = criterion(y_pred, y_train)
                # train_loss.requires_grad = True
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
                model.train()
                if loss < min_loss:
                    Best_model = model
                    min_loss = loss

    return Best_model, Total_loss

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
            x_test, y_test = x_test.float(), y_test.float()

            # compute the model output
            with torch.no_grad():
                y_pred = model(x_test)
                if y_pred.shape[-1] != 1:
                    y_pred = y_pred[:,-1]
                factors.append(y_pred)

            # delete data to release memory
            del x_test, y_test
            torch.cuda.empty_cache()
            
    # concatenate the data to get the output factors
    factors = torch.cat(factors, dim=0).cpu().detach().numpy()

    return factors
    

def train_model(save_dir: str, model_name: str, loss: str, start_year: int, end_year: int, train_window: int, test_window: int, n_class: int,
                continue_train: bool):

    # create folder to store the Model
    if not os.path.exists(f'{save_dir}/Model'):
        os.makedirs(f'{save_dir}/Model')

    # create folder to store the factor data
    if not os.path.exists(f'{save_dir}/Factor/{model_name}/{loss}'):
        os.makedirs(f'{save_dir}/Factor/{model_name}/{loss}')

    train_X, train_y = [], []

    for y in range(start_year, end_year-train_window+1, test_window):
        train_X, train_y = [], []
        test_X, test_y = [], []
        factor = pd.DataFrame()
        for train_i in range(y,y+train_window):
            f_dir = f'{save_dir}/20Day/{train_i}_image.pkl'
            l_dir = f'{save_dir}/20Day/{train_i}_label.pkl'
            X_i = read_pkl5_data(f_dir)
            y_i = read_pkl5_data(l_dir)
            y_i = y_i['return'].values
            y_i = np.where(y_i > 0, 1, 0)
            train_X.append(X_i)
            train_y.append(y_i)
        for test_i in range(y+train_window,y+train_window+test_window):
            f_dir = f'{save_dir}/20Day/{test_i}_image.pkl'
            l_dir = f'{save_dir}/20Day/{test_i}_label.pkl'
            X_i = read_pkl5_data(f_dir)
            y_i = read_pkl5_data(l_dir)
            factor = factor.append(y_i)
            y_i = y_i['return'].values
            y_i = np.where(y_i > 0, 1, 0)
            test_X.append(X_i)
            test_y.append(y_i)

        train_X = np.concatenate(train_X, axis=0)
        train_X = np.transpose(train_X, (0, 2, 1))
        train_y = np.concatenate(train_y, axis=0)
        test_X = np.concatenate(test_X, axis=0)
        test_X = np.transpose(test_X, (0, 2, 1))
        test_y = np.concatenate(test_y, axis=0)

        train_X = torch.from_numpy(train_X)
        train_y = torch.from_numpy(train_y)
        test_X = torch.from_numpy(test_X)
        test_y = torch.from_numpy(test_y)

        train_dataset = Data(train_X, train_y)
        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)

        test_dataset = Data(test_X, test_y)
        test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        model_dir = f'{save_dir}/Model/{model_name}_{loss}_{y}_{y+train_window-1}.m'

        if os.path.exists(model_dir) and not continue_train:
            Best_model = joblib.load(model_dir)
        else:
            if continue_train:
                model = joblib.load(model_dir)
                model = model.train()
            else:
                if model_name == 'CNN20d':
                    model = CNN20d()
                elif model_name == 'VGG11':
                    model = vgg11()
                elif model_name == 'VGG11_bn':
                    model = vgg11_bn()
                elif model_name == 'VGG13':
                    model = vgg13()
                elif model_name == 'VGG13_bn':
                    model = vgg13_bn()
                elif model_name == 'VGG16':
                    model = vgg16()
                elif model_name == 'VGG16_bn':
                    model = vgg16_bn()
                elif model_name == 'VGG19':
                    model = vgg19()
                elif model_name == 'VGG19_bn':
                    model = vgg19_bn()
                elif model_name == 'resnet18':
                    model = resnet18()
                elif model_name == 'resnet34':
                    model = resnet34()
                elif model_name == 'resnet50':
                    model = resnet50()
                elif model_name == 'resnet101':
                    model = resnet101()
                elif model_name == 'resnet152':
                    model = resnet152()
                elif model_name == 'vit':
                    model = vit.ViT(image_size=60, patch_size=4, num_classes=n_class, dim=1024, depth=6, heads=8, mlp_dim=2048, channels=1, dropout=0.1, emb_dropout=0.1)
                elif model_name == 'vit_1d':
                    model = vit_1d.ViT(seq_len=60, patch_size=4, num_classes=n_class, dim=1024, depth=6, heads=8, mlp_dim=2048, channels=1, dropout=0.1, emb_dropout=0.1)
                elif model_name == 'cct2':
                    model = cct.cct_2(img_size=60, n_input_channels=1, num_classes=n_class)
                elif model_name == 'cct4':
                    model = cct.cct_4(img_size=60, n_input_channels=1, num_classes=n_class)
                elif model_name == 'cct6':
                    model = cct.cct_6(img_size=60, n_input_channels=1, num_classes=n_class)
                elif model_name == 'cct7':
                    model = cct.cct_7(img_size=60, n_input_channels=1, num_classes=n_class)
                elif model_name == 'cct8':
                    model = cct.cct_8(img_size=60, n_input_channels=1, num_classes=n_class)
                elif model_name == 'cct14':
                    model = cct.cct_14(img_size=60, n_input_channels=1, num_classes=n_class)
                elif model_name == 'cct16':
                    model = cct.cct_16(img_size=60, n_input_channels=1, num_classes=n_class)
                elif model_name == 'cvt':
                    model = cvt.CvT(num_classes=n_class)
                elif model_name == 'deepvit':
                    model = deepvit.DeepViT(image_size=60, patch_size=4, num_classes=n_class, dim=1024, depth=6, heads=8, mlp_dim=2048, channels=1, dropout=0.1, emb_dropout=0.1)
                elif model_name == 'learnable_vit':
                    model = learnable_memory_vit.Adapter(
                        vit=learnable_memory_vit.ViT(image_size=60, patch_size=4, num_classes=n_class, dim=1024, depth=6, heads=8, mlp_dim=2048, channels=1, dropout=0.1, emb_dropout=0.1)
                    )
                elif model_name == 'local_vit':
                    model = local_vit.LocalViT(image_size=(64,60), patch_size=4, num_classes=n_class, dim=1024, depth=6, heads=8, mlp_dim=2048, channels=1, dropout=0.1, emb_dropout=0.1)
                elif model_name == 'maxvit':
                    model = max_vit.MaxViT(num_classes=n_class, dim=128, depth=(2,3,4), channels=1)
                # elif model_name == 'mobile_vit':
                #     model = mobile_vit.MobileViT(image_size=(64, 60), num_classes=n_class, dims=(64, 128, 256), channels=(1, 64, 128))
                elif model_name == 'parallel_vit':
                    model = parallel_vit.ViT(image_size=60, patch_size=4, num_classes=n_class, dim=1024, depth=6, heads=8, mlp_dim=2048, channels=1)
                elif model_name == 'pit':
                    model = pit.PiT(image_size=60, patch_size=4, num_classes=n_class, dim=1024, depth=(2,3,4), heads=8, mlp_dim=2048, channels=1)
                elif model_name == 'regionvit':
                    model = regionvit.RegionViT(num_classes=n_class, channels=1, local_patch_size=2, window_size=2)
                # elif model_name == 'rvt':
                #     model = rvt.RvT(image_size=60, patch_size=4, num_classes=n_class, dim=1024, depth=6, heads=8, mlp_dim=2048, channels=1)
                # elif model_name == 'sep_vit':
                    # model = sep_vit.SepViT(num_classes=n_class, dim=128, depth=(2,3,4), channels=1, heads=8)
                # elif model_name == 'simple_flash_attn_vit':
                #     model = simple_flash_attn_vit.SimpleViT(image_size=60, patch_size=4, num_classes=n_class, dim=1024, depth=6, heads=8, mlp_dim=2048, channels=1)
                elif model_name == 'simple_vit':
                    model = simple_vit.SimpleViT(image_size=(64,60), patch_size=4, num_classes=n_class, dim=1024, depth=6, heads=8, mlp_dim=2048, channels=1)
                elif model_name == 'simple_vit_1d':
                    model = simple_vit_1d.SimpleViT(seq_len=60, patch_size=4, num_classes=n_class, dim=1024, depth=6, heads=8, mlp_dim=2048, channels=1)
                elif model_name == 'simple_vit_with_patch_dropout':
                    model = simple_vit_with_patch_dropout.SimpleViT(image_size=(64,60), patch_size=4, num_classes=n_class, dim=1024, depth=6, heads=8, mlp_dim=2048, channels=1)
                # elif model_name == 'twins_svt':
                #     model = twins_svt.TwinsSVT(num_classes=n_class)
                elif model_name == 'vit_for_small_dataset':
                    model = vit_for_small_dataset.ViT(image_size=(64,60), patch_size=4, num_classes=n_class, dim=1024, depth=6, heads=8, mlp_dim=2048, channels=1)
                else:
                    raise ValueError(f'The parameter model_name is invalid, get {model_name}')
            
            if loss == 'BCE':
                loss_func = BCE()
            elif loss == 'IC':
                loss_func = IC_loss()
            elif loss == 'CCC':
                loss_func = CCC_loss()
            # elif loss == 'ListMLE':
            #     loss_func = ListMLE()
            # elif loss == 'ListNet':
            #     loss_func = ListNet()
            # elif loss == 'Closs':
            #     loss_func = Closs()
            # elif loss == 'Closs_explained':
            #     loss_func = Closs_explained()
            # elif loss == 'Closs_sigmoid':
            #     loss_func = Closs_sigmoid()
            # elif loss == 'Lloss':
            #     loss_func = Lloss()
            else:
                raise ValueError(f'The parameter loss is invalid, get {loss}')

            model = model.to(device)
            Best_model, _ = train(model, train_dataloader, loss_func)
            joblib.dump(Best_model, model_dir)

        pred_y = get_factor(Best_model, test_dataloader)
        factor['factor'] = pred_y
        dateLst = np.sort(pd.unique(factor['date']))

        # load the training dataset
        with trange(len(dateLst)) as date_bar:
            for i in date_bar:
                date_i = dateLst[i]
                date_bar.set_description(f'Saving factor data on date {date_i}')

                factor_i = factor.loc[factor['date'] == date_i]
                factor_i.to_csv(f'{save_dir}/Factor/{model_name}/{loss}/{date_i}.csv')


def backtest(save_dir: str, model_name: str, loss: str, start_date: int, end_date: int, thres: int, sign: int=1, method: str='topN'):

    # create folder to store the factor data
    if not os.path.exists(f'{save_dir}/Report'):
        os.makedirs(f'{save_dir}/Report')
    
    factorLst = os.listdir(f'{save_dir}/Factor/{model_name}/{loss}')
    factorLst = [dir for dir in factorLst if dir >= f'{start_date}.csv' and dir <= f'{end_date}.csv']

    factor = pd.DataFrame()
    # load the training dataset
    with trange(len(factorLst)) as date_bar:
        for i in date_bar:
            date_i = factorLst[i]
            date_bar.set_description(f'Loading factor data from file {date_i}')
            factor_i = pd.read_csv(f'{save_dir}/Factor/{model_name}/{loss}/{date_i}', index_col=0)
            factor = factor.append(factor_i)

    factor.reset_index(drop=True, inplace=True)

    return_df = factor.loc[:,['id', 'date', 'return']].copy()
    df = factor.loc[:,['date', 'factor', 'return']].copy()

    # get the return of benchmark
    bench_df = return_df.groupby(by='date')['return'].mean().reset_index()
    bench_df.rename(columns={'return': 'benchmark'}, inplace=True)

    # calculate return of top N stocks based on their predicted score
    def topN(x):
        score = x['factor'].values
        ret = x['return'].values
        if sign == 1:
            ids = np.argsort(score)[::-1]
        elif sign == -1:
            ids = np.argsort(score)
        else:
            raise ValueError(f'The parameter sign should be -1/1, get {sign} instead')
        if method == 'topN':
            return np.nanmean(ret[ids[:thres]])
        elif method == 'Percent':
            return np.nanmean(ret[ids[:int(ret.shape[0]*thres)]])
        else:
            raise ValueError(f'The parameter should be topN/Percent, get {method} instead')

    # calculate the return of long portfolio
    df = df.groupby('date').apply(topN)
    df.name = 'return'
    df = df.to_frame()
    df.reset_index(inplace=True)
    portfolio = pd.merge(df, bench_df, on='date')
    portfolio['date'] = portfolio['date'].apply(lambda date_i: pd.Timestamp(int(str(date_i)[:4]), int(str(date_i)[4:6]), int(str(date_i)[6:])))
    portfolio.set_index('date', inplace=True)
    print(portfolio)

    # create the report under the path
    report_dir = f'{save_dir}/Report/{model_name}_{loss}_test_{start_date}_{end_date}.html'

    qs.reports.html(portfolio['return'], portfolio['benchmark'],
        title=f'Report of long-short portfolio with factor predicted by MTL',
        output=report_dir)
    
    print('Report saved in %s' % (report_dir))