import os
import sys
import pickle
import numpy as np
import pandas as pd
import quantstats as qs
import cvxpy as cp
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

# The backtest structure
class BackTest(object):
    def __init__(self, save_dir: str, stocklist_dir: str, trDays_dir: str, start_date: int, end_date: int) -> None:
        self.save_dir = save_dir
        self.stocklist_dir = stocklist_dir
        self.trDays_dir = trDays_dir
        self.start_date = start_date
        self.end_date = end_date

    # load the required data
    def init_data(self):
        print('Loading the stock list data ...')
        self.stockDF = read_pkl(self.stocklist_dir)
        print('Loading the trading dates data ...')
        self.trDays = pd.read_hdf(self.trDays_dir)
        self.trDays = self.trDays.values
        self.trDays = self.trDays[(self.trDays >= self.start_date) & (self.trDays <= self.end_date)]

    # processing the strategy to get daily return data
    def forward(self):
        signal = pd.read_csv('%s/Signal/RPS_signal_%s_%s.csv' % (self.save_dir, self.start_date, self.end_date))
        # merge factor data with stock data
        stock = self.stockDF.loc[:,['id', 'date', 'ret', 'hs300_PCT_CHG']]
        stock['ret'] = stock.groupby('id')['ret'].shift(-1) / 100
        stock['hs300_PCT_CHG'] = stock.groupby('id')['hs300_PCT_CHG'].shift(-1) / 100
        stock = stock.loc[(stock['date'] >= self.start_date) & (stock['date'] <= self.end_date)]
        stock = pd.merge(signal, stock, on=['id', 'date'], how='left')
        stock = stock.dropna(subset=['ret', 'hs300_PCT_CHG'])
        dateLst = pd.unique(stock['date'].values)
        dateLst = np.sort(dateLst)
        # get daily return
        ret_signal, ret_bench = [], []
        with trange(len(dateLst)) as date_bar:
            for i in date_bar:
                date_i = dateLst[i]
                date_bar.set_description('Processing data on date %s' % date_i)
                df_i = stock.loc[stock['date'] == date_i]
                # choose return of hs300 index as the return of benchmark
                ret_bench.append(df_i['hs300_PCT_CHG'].iloc[0])
                if df_i['signal'].sum() == 0:
                    ret_signal.append(0)
                else:
                    ret_signal.append((df_i['ret'] * df_i['signal']).sum() / df_i['signal'].sum())

            # make long short portfolio
            dateLst = [pd.Timestamp(int(str(date_i)[:4]), int(str(date_i)[4:6]), int(str(date_i)[6:])) for date_i in dateLst]
            self.portfolioDF = pd.DataFrame({'date': dateLst, 'yieldRate': ret_signal, 'bench': ret_bench})
            self.portfolioDF.set_index(['date'],inplace=True)

            print(self.portfolioDF)
            # create the folder to store the report
            if not os.path.exists('%s/report' % self.save_dir):
                os.makedirs('%s/report' % self.save_dir)
            # create the report under the path
            report_dir = '%s/report/%s_%s.html'\
                %(self.save_dir, self.start_date, self.end_date)
            qs.reports.html(self.portfolioDF['yieldRate'], self.portfolioDF['bench'],
                title='Report of long-short portfolio with signal by RPS',
                output=report_dir)
            print('Report saved in %s' % (report_dir))


# The backtest structure
class BackTest_RPS(object):
    def __init__(self, save_dir: str, stocklist_dir: str, trDays_dir: str, start_date: int, end_date: int,
        thres=None, window=None, period=None) -> None:

        self.save_dir = save_dir
        self.stocklist_dir = stocklist_dir
        self.trDays_dir = trDays_dir
        self.start_date = start_date
        self.end_date = end_date
        self.thres = thres
        self.window = window
        self.period = period

    # load the required data
    def init_data(self):
        print('Loading the stock list data ...')
        self.stockDF = read_pkl(self.stocklist_dir)
        print('Loading the signal data ...')
        if len(self.window) == 1:
            self.Signal = pd.DataFrame()
            dirLst = os.listdir('%s/Factors/RPS_%s' % (self.save_dir, self.window[0]))
            dirLst = [dir for dir in dirLst if int(dir[:8]) >= self.start_date and int(dir[:8]) <= self.end_date]
            with trange(len(dirLst)) as dir_bar:  
                for i in dir_bar:
                    dir_i = dirLst[i]
                    dir_bar.set_description('Preprocessing file %s' % dir_i)
                    # get the factor data when needed
                    df_i = pd.read_csv('%s/Factors/RPS_%s/%s' % (self.save_dir, self.window[0], dir_i), index_col=[0])
                    self.Signal = self.Signal.append(df_i)
            self.Signal.reset_index(drop=True, inplace=True)
            self.Signal = self.Signal[['id', 'date', 'RPS_%s' % self.window[0]]]
        else:
            signalLst = []
            for w in self.window:
                temp = pd.DataFrame()
                dirLst = os.listdir('%s/Factors/RPS_%s' % (self.save_dir, w))
                dirLst = [dir for dir in dirLst if int(dir[:8]) >= self.start_date and int(dir[:8]) <= self.end_date]
                with trange(len(dirLst)) as dir_bar:  
                    for i in dir_bar:
                        dir_i = dirLst[i]
                        dir_bar.set_description('Preprocessing file %s' % dir_i)
                        # get the factor data when needed
                        df_i = pd.read_csv('%s/Factors/RPS_%s/%s' % (self.save_dir, w, dir_i), index_col=[0])
                        temp = temp.append(df_i)
                temp.reset_index(drop=True, inplace=True)
                temp = temp[['id', 'date', 'RPS_%s' % w]]
                signalLst.append(temp)
            self.Signal = reduce(lambda x, y: pd.merge(x, y, on=['id', 'date'], how='left'), signalLst)

    # processing the strategy to get daily return data
    def forward(self):
        # merge factor data with stock data
        stock = self.stockDF.loc[:,['id', 'date', 'close', 'hs300_CLOSE']]
        stock['ret'] = stock.groupby('id')['close'].pct_change(self.period).shift(-self.period)
        stock['hs300_PCT_CHG'] = stock.groupby('id')['hs300_CLOSE'].pct_change(self.period).shift(-self.period)
        stock = pd.merge(self.Signal, stock, on=['id', 'date'], how='inner')
        stock = stock.dropna(subset=['ret', 'hs300_PCT_CHG'])
        dateLst = pd.unique(stock['date'].values)
        dateLst = np.sort(dateLst)
        dateLst = [dateLst[i] for i in range(0,len(dateLst),self.period)]
        # get daily return
        ret_signal, ret_bench = [], []
        last_portfolio = pd.DataFrame()
        with trange(len(dateLst)) as date_bar:
            for i in date_bar:
                date_i = dateLst[i]
                date_bar.set_description('Processing data on date %s' % date_i)
                df_i = stock.loc[stock['date'] == date_i]
                # choose return of hs300 index as the return of benchmark
                ret_bench.append(df_i['hs300_PCT_CHG'].iloc[0])
                temp = df_i.copy()
                for w in self.window:
                    temp = temp.loc[df_i['RPS_%s' % w] <= self.thres]
                if temp.empty:
                    if last_portfolio.empty:
                        ret_signal.append(0)
                        continue
                    else:
                        df_i = pd.merge(df_i, last_portfolio, on='id', how='right')
                else:
                    df_i = temp.copy()
                ret_signal.append(df_i['ret'].sum() / df_i.shape[0])
                last_portfolio = df_i['id'].to_frame()

        # make long short portfolio
        dateLst = [pd.Timestamp(int(str(date_i)[:4]), int(str(date_i)[4:6]), int(str(date_i)[6:])) for date_i in dateLst]
        self.portfolioDF = pd.DataFrame({'date': dateLst, 'yieldRate': ret_signal, 'bench': ret_bench})
        self.portfolioDF.set_index(['date'],inplace=True)

        print(self.portfolioDF)
        # create the folder to store the report
        if not os.path.exists('%s/report' % self.save_dir):
            os.makedirs('%s/report' % self.save_dir)
        # create the report under the path
        report_dir = '%s/report/%s_%s_%s_%s.html'\
            %(self.save_dir, self.start_date, self.end_date, self.thres, self.period)
        qs.reports.html(self.portfolioDF['yieldRate'], self.portfolioDF['bench'],
            title='Report of long-short portfolio with signal by RPS',
            output=report_dir)
        print('Report saved in %s' % (report_dir))