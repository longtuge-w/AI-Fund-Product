import sys
import os
import pickle
import joblib
import numpy as np 
import pandas as pd 
import quantstats as qs
import warnings


class backtest(object):
    def __init__(self,portfolio_df,test_x,test_y,timeID,model_dir,stocklist_dir,save_dir,numWindow,
        weight='equal',bench='zz500',numStock=100,trade_fee=0,threshold=0.0001,mission='regression',
        data_type='Dataframe'):
        self.portfolio_df = portfolio_df
        self.test_x = test_x
        self.test_y = test_y
        self.timeID = timeID
        self.model_dir = model_dir
        self.stocklist_dir = stocklist_dir
        self.save_dir = save_dir
        self.numStock = numStock
        self.weight = weight
        self.bench = bench
        self.numWindow = numWindow
        self.trade_fee = trade_fee
        self.threshold = threshold
        self.mission = mission
        self.cash = 1
        self.data_type = data_type
        """
        model: list
        a list contianing pre-trained models, if the length of this list is 1, use the model all the time. Otherwise, 
        iterating the list and apply the model to prediction
        portfolio_df: pd.DataFrame
        a dataframe with columns "date" and "id", indicating for each date, the stock pool contains which stocks
        test_x: list
        a list containing the test data for features
        test_y: list
        a list containing the test data for stock returns
        stocklist_dir: str
        path where stock daily data is saved
        numStock: int
        the number of stock in the original portfolio
        weight: str
        the method determining the weight of the portfolio, which should be equal/market_value
        bench: str
        the benchmark we choose, should be either hs300 or zz500
        numWindow: int
        # of window, representing the value of n for T+n strategy
        trade_fee: float
        the rate of transaction fee
        threshold: float
        above which should we do the buy or sell behaviour
        mission: str
        the mission model does, should be either classification or regression
        data_type: str
        the type of the training and test data, should be 'Dataframe'/'ndarray'
        """

    def set_numStock(self,stocklist_dir):
        self.stocklist_dir = stocklist_dir

    def set_save_dir(self,save_dir):
        self.save_dir = save_dir

    def set_numStock(self,numStock):
        self.numStock = numStock

    def set_weight(self,weight):
        self.weight = weight

    def set_numWindow(self,numWindow):
        self.numWindow = numWindow

    def set_trade_fee(self,trade_fee):
        self.trade_fee = trade_fee

    def set_threshold(self,threshold):
        self.threshold = threshold

    def set_data_type(self,data_type):
        self.data_type = data_type


    # predict the yieldrate based on data and pre-trained model
    def predict(self,mission,modelTrained,testX):
        if self.data_type == 'Dataframe':
            if mission == 'classification':		# some model do the classification 
                if hasattr(modelTrained,'predict_proba'):
                    pred_Y = modelTrained.predict_proba(testX)[:,1]
                elif hasattr(modelTrained,'_predict_proba_lr'):
                    pred_Y = modelTrained._predict_proba_lr(testX)[:,1]
                else:
                    raise AttributeError('This estimator has no suitable attribute to predict the probability for each class')
            elif mission == 'regression':		# some model just do the regression no probability 
                pred_Y = modelTrained.predict(testX)
            else:
                raise ValueError('the mission of model should be either classification or regression instead of %s'%(mission))
        elif self.data_type == 'ndarray':
            if mission == 'classification':		# some model do the classification 
                if hasattr(modelTrained,'predict_proba'):
                    pred_Y = modelTrained.predict_proba(testX.values)[:,1]
                elif hasattr(modelTrained,'_predict_proba_lr'):
                    pred_Y = modelTrained._predict_proba_lr(testX.values)[:,1]
                else:
                    raise AttributeError('This estimator has no suitable attribute to predict the probability for each class')
            elif mission == 'regression':		# some model just do the regression no probability 
                pred_Y = modelTrained.predict(testX.values)
            else:
                raise ValueError('the mission of model should be either classification or regression instead of %s'%(mission))
        else:
            raise ValueError('The value of data_type should be Dataframe/ndarray instead of %s'%(self.data_type))
        return pred_Y.flatten()


    # get the information of the final portfolio
    def get_portfolio(self):
        ret_lst, bench_lst, date_lst = [], [], []
        self.last_P,self.current_P,self.stock_pool,stockData = pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
        stockpool_lst = pd.unique(self.portfolio_df.id)

        if self.bench == 'hs300':
            bench_col = 'hs300_CLOSE'
        elif self.bench == 'zz500':
            bench_col = 'zz500_CLOSE'
        else:
            raise ValueError('The value of benchmark should be either hs300 or zz500 instead of %s'%(self.bench))

        f = open(self.stocklist_dir,'rb')
        stockLst = pickle.load(f)
        for stockDF in stockLst:
            stock_id = stockDF.id.iloc[0]
            if stock_id not in stockpool_lst:
                continue
            stockDF.tdate = stockDF.tdate.astype(str)
            stockDF = stockDF.loc[stockDF.tdate.isin(self.timeID)]\
                [['tdate','id','open','high','low','close','volume','market_value']+[bench_col]].copy()
            stockData = stockData.append(stockDF)
            print('stock %s has been uploaded'%(stock_id))
        stockData['limit'] = np.where(((stockData['close']==stockData['open'])&(stockData['open']==stockData['high'])\
            &(stockData['high']==stockData['low']))|(stockData['volume']==0),1,0)
        stockData.rename(columns={'tdate':'date'},inplace=True)
        stockData.date = stockData.date.astype(np.int32)
        self.portfolio_df = pd.merge(self.portfolio_df,stockData,on=['id','date'],how='left')
        self.portfolio_df.set_index('date',inplace=True)

        # find the corresponding model and test data
        print('Predicting stock returns...')
        old_model_index = -1
        for i in range(len(self.timeID)-1):
            # load the data
            timeID_i,timeID_i1 = self.timeID[i],self.timeID[i+1]
            benchRate_i = self.portfolio_df[bench_col].loc[timeID_i1].iloc[0]/self.portfolio_df[bench_col].loc[timeID_i].iloc[0]-1
            portfolio_i = self.portfolio_df.loc[timeID_i].copy()
            stockID_i = portfolio_i.id.tolist()
            marketValue_i = portfolio_i.market_value
            limit_i = portfolio_i.limit
            data_x_i = self.test_x[i]
            data_y_i = self.test_y[i]

            model_lst = sorted(os.listdir(self.model_dir))
            modelDate_lst = [i[:8] for i in model_lst]
            model_index = len(modelDate_lst+[timeID_i])-sorted(modelDate_lst+[timeID_i],reverse=True).index(timeID_i)-2
            print('load model %s'%(model_lst[model_index]))
            if model_index != old_model_index:
                model = joblib.load(r'%s\%s'%(self.model_dir,model_lst[model_index]))
                old_model_index = model_index

            predY_i = np.array(self.predict(self.mission,model,data_x_i))
            # get the return of the portfolio, and set the return of all stocks in stock pool as the bench return
            portfolio_index = predY_i.argsort()[-self.numStock:]
            bench_lst.append(benchRate_i)
            # current portfolio, we choose the stocks with highest factor score
            self.current_P['stockID'] = np.array(stockID_i)[portfolio_index]
            self.current_P['limit'] = np.array(limit_i)[portfolio_index]
            # stock pool
            self.stock_pool['stockID'] = np.array(stockID_i)
            self.stock_pool['return'] = data_y_i+1
            # add column market value when using market value weighted method
            if self.weight == 'market_value':
                self.current_P['market_value'] = np.array(marketValue_i)[portfolio_index]
                self.stock_pool['market_value'] = np.array(marketValue_i)

            # trade the stock
            ret_i = self.trade()
            ret_lst.append(ret_i)

            date_i = timeID_i1
            date_lst.append(date_i)
            print('The trade process on %s has been executed'%(date_i))
        self.start_date,self.end_date = date_lst[0],date_lst[-1]
        date_lst = [pd.Timestamp(int(str(date_i)[:4]),int(str(date_i)[4:6]),int(str(date_i)[6:])) for date_i in date_lst]
        self.portfolioDF = pd.DataFrame({'date':date_lst,'yieldRate':ret_lst,'bench':bench_lst})
        self.portfolioDF.set_index(['date'],inplace=True)
        print('Finish predicting')


    def trade(self):
        # divide weights for the remaining stocks in the portfolio
        if self.weight == 'equal':
            self.current_P['weight'] = 1/self.current_P.shape[0]

        elif self.weight == 'market_value':
            self.current_P['weight'] = self.current_P.market_value/self.current_P.market_value.sum()
            self.current_P.drop(columns=['market_value'],inplace=True)

        # initial the last_P for the first trading day
        if self.last_P.empty:
            self.last_P['stockID'] = self.current_P['stockID']
            self.last_P['last_weight'] = 0

        # merge the last portfolio and the current portfolio
        merge_P_i = pd.merge(self.current_P,self.last_P,how='outer',on='stockID')
        merge_P_i.fillna(0,inplace=True)
        merge_P_i['change'] = merge_P_i.weight - merge_P_i.last_weight
        # delete the stocks that cannot be traded
        merge_P_i = merge_P_i.loc[merge_P_i.limit==0]
        limit_P_i = merge_P_i.loc[merge_P_i.limit!=0]
        limit_P_i = limit_P_i.loc[limit_P_i.last_weight!=0]
        # hold the stock when the weight we wanna buy is too low, otherwise buy or sell it
        hold_P_i = merge_P_i.loc[(abs(merge_P_i.change)<=self.threshold)]
        buy_P_i = merge_P_i.loc[merge_P_i.change>self.threshold]
        sell_P_i = merge_P_i.loc[merge_P_i.change<-self.threshold]
        hold_P_i = hold_P_i.append(limit_P_i)
        hold_P_i.weight = hold_P_i.last_weight
        # the cash we get after selling stocks
        self.cash += -sell_P_i.change.sum()*(1-self.trade_fee)
        sell_P_i = sell_P_i.loc[sell_P_i.weight!=0]
        sell_P_i = pd.merge(sell_P_i,self.stock_pool,on='stockID',how='left')
        buy_P_i = pd.merge(buy_P_i,self.stock_pool,on='stockID',how='left')
        hold_P_i = pd.merge(hold_P_i,self.stock_pool,on='stockID',how='left')

        if self.weight == 'equal':
            buy_P_i['change'] = self.cash/buy_P_i.shape[0]
            buy_P_i['weight'] = buy_P_i['last_weight']+buy_P_i['change']

        elif self.weight == 'market_value':
            buy_P_i['change'] = self.cash*buy_P_i.market_value/buy_P_i.market_value.sum()
            buy_P_i['weight'] = buy_P_i['last_weight']+buy_P_i['change']

        # buy the stocks
        self.cash -= buy_P_i.change.sum()*(1+self.trade_fee)
        # get the portfolio
        hold_P_i = hold_P_i.append(buy_P_i).append(sell_P_i)
        hold_P_i.reset_index(inplace=True,drop=True)
        hold_P_i['weight'] *= hold_P_i['return']
        total_cash_i = hold_P_i.weight.sum()+self.cash
        ret_i = total_cash_i-1
        hold_P_i['weight'] = hold_P_i['weight']/hold_P_i['weight'].sum()
        self.last_P = pd.DataFrame({'stockID':hold_P_i.stockID,'last_weight':hold_P_i.weight})
        self.current_P,self.stock_pool = pd.DataFrame(),pd.DataFrame()

        return ret_i


    def sharpRatio(self):
        return qs.stats.sharpe(self.portfolioDF['yieldRate'])


    def infoRatio(self):
        return qs.stats.information_ratio(self.portfolioDF['yieldRate'],self.portfolioDF['bench'])


    def CalmarRatio(self):
        return qs.stats.calmar(self.portfolioDF['yieldRate'])


    def create_report(self):
        print(self.portfolioDF)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        report_dir = r'%s\%s_T+%s_%s_%s_Top%s.html'\
            %(self.save_dir,self.weight,self.numWindow,self.start_date,self.end_date,self.numStock)
        qs.reports.html(self.portfolioDF['yieldRate'],self.portfolioDF['bench'],
            title='T+%s strategy with %s weight portfolio'%(self.numWindow,self.weight),
            output=report_dir)
        print('Report saved in %s'%(report_dir))


def main(portfolio_df,test_x,test_y,timeID,model_dir,stocklist_dir,save_dir,numWindow,weight,bench,trade_fee,threshold,mission,data_type='Dataframe'):
    # used to ignore the warnings
    warnings.filterwarnings("ignore")

    p = backtest(portfolio_df,test_x,test_y,timeID,model_dir,stocklist_dir,save_dir,numWindow
        ,weight=weight,bench=bench,trade_fee=trade_fee,threshold=threshold,mission=mission
        ,data_type=data_type)

    '''use quanstats to analyze the performance'''
    p.get_portfolio()

    # print(p.sharpRatio())
    # print(p.infoRatio())
    # print(p.CalmarRatio())
    p.create_report()

# if __name__ == '__main__':

    # """
    # doing backtest
    # """
    # startDay = 20190118
    # endDay = 20191231
    # stocklist_dir = r'D:\实习\博时基金\东方证券\东方金工组合优化函数\测试\归因可视化\stockList_all_4394_20100104_20210531.pkl'
    # weight = 'equal'
    # bench = 'zz500'
    # numWindow = 5
    # trade_fee = 0.003
    # threshold = 0.0001
    # save_dir = r'.\backtest_report\zz500'
    # model_dir = r'.\ModelHouse\zz500\unopt\20181228_5_30_240.m'
    # portfolio_dir = r'.\DataHouse\zz500\portfolio'
    # data_dir = r'.\DataHouse\zz500\train_test'
    # startData_dir = '20190118_5_30.pkl'
    # endData_dir = '20191231_5_30.pkl'

    # test_x,test_y = [],[]
    # data_lst = os.listdir(data_dir)
    # start_index = data_lst.index(startData_dir)
    # end_index = data_lst.index(endData_dir)
    # for i in range(start_index,end_index+1,numWindow):
    #     print('reading test data %s'%(data_lst[i]))
    #     f = open(r'%s\%s'%(data_dir,data_lst[i]),'rb')
    #     test_x_i,test_y_i = pickle.load(f)
    #     test_x_i,test_y_i = np.array(test_x_i),np.array(test_y_i)
    #     test_x_i,test_y_i = torch.tensor(test_x_i,dtype=dtype),torch.tensor(test_y_i,dtype=dtype)
    #     test_x.append(test_x_i)
    #     test_y.append(test_y_i)

    # portfolioDF = pd.DataFrame()
    # portfolio_lst = os.listdir(portfolio_dir)
    # start_index = portfolio_lst.index(startData_dir)
    # end_index = portfolio_lst.index(endData_dir)
    # for i in range(start_index,end_index+1,numWindow):
    #     print('reading portfolio data %s'%(portfolio_lst[i]))
    #     f = open(r'%s\%s'%(portfolio_dir,portfolio_lst[i]),'rb')
    #     portfolio = pickle.load(f)
    #     portfolioDF = portfolioDF.append(portfolio)

    # main(portfolioDF,test_x,test_y,model_dir,stocklist_dir,save_dir,numWindow,weight,bench,trade_fee,threshold)
