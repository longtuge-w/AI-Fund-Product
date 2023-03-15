import pandas as pd
import numpy as np
import statsmodels.api as sm
import cvxpy as cp
import time
import pickle
import os
from DataInfo import DataInfo


class getData(DataInfo):
    def __init__(self):
        DataInfo.__init__(self)
        self.start = None # start date
        self.end = None # end date
        self.current = None # current date
        self.Time_data = None # the trading date series from start date to end date
        self.cov_data = None # store the covariance data
        self.opt_merge_data = None # the data used to do optimization
        self.stock = None # store the daily data of stocks
        self.merge_data = None # merge data of optimization portfolio
        self.merge_data_equal = None # merge data of equal weight portfolio
        self.merge_data_factor = None # merge data of factor weighted portfolio
        self.FacRet = {} # factor return data
        self.Industry_weight = {} # market percentage data
        self.Portfolio = {} # weights of stock in opimization portfolio
        self.Portfolio_equal = {} # weights of stock in equal weight portfolio
        self.Portfolio_factor = {} # weights of stock in factor weighted portfolio
        self.FacVar = {} # variance of stock return
        # the following data structures are used when no optimization is needed
        self.Portfolio_lst = [{}]*10
        self.merge_data_lst = None
        self.portfolio_lst = [i for i in [self.Portfolio_dir_1,self.Portfolio_dir_2,self.Portfolio_dir_3,self.Portfolio_dir_4
                    ,self.Portfolio_dir_5,self.Portfolio_dir_6,self.Portfolio_dir_7,self.Portfolio_dir_8,
                    self.Portfolio_dir_9,self.Portfolio_dir_10] if i is not None]
        self.portfolio_lst_lst = []
        self.file_lst = [0]*len(self.portfolio_lst)


    def set_startDate(self,startDate):
        self.start_date = startDate


    def set_endDate(self,endDate):
        self.end_date = endDate


    def set_window(self,window):
        self.window = window

    
    def set_format(self,dataFormat):
        self.format = dataFormat


    def set_optimization(self,optimization):
        self.optimization = optimization

    
    def set_optCode(self,optCode):
        self.OptCode = optCode


    def set_method(self,method):
        self.method = method


    def set_methodValue(self,methodValue):
        self.method_value = methodValue


    # turn the format of one column into xxxxxx.SZ
    def get_stock_code(self,df,column='id'):
        df[column] = df[column].astype(str)
        df[column] = df[column].apply(lambda x: x.zfill(6))
        df[column] = df[column].apply(lambda x: x + '.SH' if x[0] == '6' else x + '.SZ')
        return df


    # shorten the name of industry and style factors
    def get_factor_name(self,df,column='factor'):
        df[column] = df[column].apply(lambda x: x[6:])
        return df


    # save the pkl data
    def save_pkl(self,path_lst,dict_lst):
        if len(path_lst) != len(dict_lst):
            raise ValueError('The length of two list must be equal')
        for i in range(len(path_lst)):
            with open(path_lst[i], "wb") as f:
                pickle.dump(dict_lst[i], f)
                f.close()
            

    # create files
    def makedir(self,path):
        folder = os.path.exists(path)
        if not folder:
            os.makedirs(path)


    # change the path to read different data files
    def change_portfolio_dir(self,n):
        for i in range(len(self.portfolio_lst_lst)):
            self.file_lst[i] = r'%s\%s'%(self.portfolio_lst[i],self.portfolio_lst_lst[i][n])
        

    # get the time series data
    def get_time_series(self):

        # can also use database
        Time_pkl = open(self.Time_dir,'rb')
        Time_lst = pickle.load(Time_pkl).tolist()

        # if the chosen start date is not trading day, then choose the latest following trading day
        if Time_lst.count(self.start_date):
            start_index = Time_lst.index(self.start_date)
            print('This code will read the data from %s'%(self.start_date))
        else:
            Time_lst.append(self.start_date)
            Time_lst.sort()
            start_index = Time_lst.index(self.start_date)
            Time_lst.remove(self.start_date)
            print('%s is not a trading day. Taking %s instead...'%(self.start_date,Time_lst[start_index]))
            if self.start_date == self.end_date:
                return [Time_lst[start_index-self.window:start_index+1]]

        # if the chosen end date is not trading day, then choose the latest following trading day
        if Time_lst.count(self.end_date):
            end_index = Time_lst.index(self.end_date)+1
            print('This code will read the data up to %s'%(self.end_date))
        else:
            Time_lst.append(self.end_date)
            Time_lst.sort()
            end_index = Time_lst.index(self.end_date)
            Time_lst.remove(self.end_date)
            print('%s is not a trading day. Taking %s instead...'%(self.end_date,Time_lst[end_index]))

        return Time_lst[start_index-self.window:end_index]


    # get the list containing the required files
    def get_fileLst(self,path,start,end):
        
        path_lst = sorted(os.listdir(path))
        start_path = [file for file in path_lst if start in file][0]
        end_path = [file for file in path_lst if end in file][0]
        start_index = path_lst.index(start_path)
        end_index = path_lst.index(end_path)

        return path_lst[start_index:end_index+1]


    def get_opt_merge_data(self,Bench_dir,Factor_dir):
        # can also use database
        # factor score data
        Factor_df = pd.read_csv(r'%s\%s'%(self.Factor_dir,Factor_dir),names=['id','factor'])
        Factor_df = self.get_stock_code(Factor_df)
        # benchmark data
        Bench_df = pd.read_csv(r'%s\%s'%(self.Bench_dir,Bench_dir))
        Bench_df = pd.DataFrame({'id':Bench_df['S_CON_WINDCODE'],'i_weight':Bench_df['WEIGHT']})
        # Barra exposure data
        BarraExp_df = pd.DataFrame()
        f = open(self.Barra_dir,'rb')
        data = pickle.load(f)
        for i in data:
            BarraExp_df = BarraExp_df.append(i.loc[i['tdate']==self.start])
        BarraExp_df = self.get_stock_code(BarraExp_df)
        del BarraExp_df['tdate']
        # merge the above data
        opt_merge_data = pd.merge(Factor_df,BarraExp_df,on='id',how='inner')
        opt_merge_data = pd.merge(opt_merge_data,Bench_df,on='id',how='inner')
        self.makedir(self.Save_dir+r'.\weight')
        try:
            # the weights of stock on the last trading day, if not exists, use the benchmark weights
            weight = pd.read_csv(self.Save_dir+r'\weight\%sweight.csv' % (self.current))
            opt_merge_data = pd.merge(opt_merge_data,weight,on='id',how='left')
            opt_merge_data = opt_merge_data.fillna(0)
        except:
            opt_merge_data['w_last'] = opt_merge_data['i_weight']
        return opt_merge_data


    # drop some stocks with small weight
    def get_portfolio(self):
        Portfolio = pd.read_csv(self.Save_dir+r'\cvx_portfolio\%sportfolio.csv' % (self.end))
        Portfolio['id'] = self.opt_merge_data['id']
        if not (self.method_value == None or isinstance(self.method_value,float)):
            raise TypeError('The value of \"self.method_value\" should be None or float')
        if self.method == 'value': # drop all stocks whose value is smaller than one specific value
            if self.method_value == None:
                Portfolio = Portfolio.loc[Portfolio['w_opt']>=0.0001]
            else:
                Portfolio = Portfolio.loc[Portfolio['w_opt']>=self.method_value]
        elif self.method == 'percentage': # drop all stocks whose rank is smaller
            if self.method_value == None:
                Portfolio = Portfolio.loc[Portfolio['w_opt']>=np.percentile(0.8)]
            else:
                Portfolio = Portfolio.loc[Portfolio['w_opt']>=np.percentile(self.method_value)]
        else:
            raise ValueError('The value of \"method\" should be \"value\" or \"percentage\"')
        # update weight after droping some stocks
        total_weight = Portfolio['w_opt'].sum()
        Portfolio['w_opt'] /= total_weight
        Portfolio.rename(columns={'w_opt':'w_last'},inplace=True)
        Portfolio.to_csv(self.Save_dir+r'\weight\%s_weight.csv' % (self.end),index=False)
        Portfolio.rename(columns={'w_last':'w_opt'},inplace=True)

        return Portfolio


    def get_portfolio_equal_weight(self,Factor_dir):
        Portfolio = pd.read_csv(r'%s\%s'%(self.Factor_dir,Factor_dir),names=['id','factor'])
        Portfolio = self.get_stock_code(Portfolio)
        if not (self.method_value == None or isinstance(self.method_value,int) or isinstance(self.method_value,float)):
            raise TypeError('The value of \"self.method_value\" should be None or int')
        if self.method == 'value':
            if self.method_value == None:
                Portfolio = Portfolio.head(100)
            else:
                Portfolio = Portfolio.head(self.method_value)
        elif self.method == 'percentage':
            if self.method_value == None:
                Portfolio = Portfolio.head(int(Portfolio.shape[0]*0.2))
            else:
                Portfolio = Portfolio.head(int(Portfolio.shape[0]*self.method_value))
        else:
            raise ValueError('The value of \"method\" should be \"value\" or \"percentage\"')
        Portfolio['w_equal_weight'] = 1/Portfolio.shape[0]
        del Portfolio['factor']

        return Portfolio


    def get_portfolio_factor_weight(self,Factor_dir):
        Portfolio = pd.read_csv(r'%s\%s'%(self.Factor_dir,Factor_dir),names=['id','factor'])
        Portfolio = self.get_stock_code(Portfolio)
        if not (self.method_value == None or isinstance(self.method_value,int) or isinstance(self.method_value,float)):
            raise TypeError('The value of \"self.method_value\" should be None or int')
        if self.method == 'value':
            if self.method_value == None:
                Portfolio = Portfolio.head(100)
            else:
                Portfolio = Portfolio.head(self.method_value)
        elif self.method == 'percentage':
            if self.method_value == None:
                Portfolio = Portfolio.head(int(Portfolio.shape[0]*0.2))
            else:
                Portfolio = Portfolio.head(int(Portfolio.shape[0]*self.method_value))
        else:
            raise ValueError('The value of \"method\" should be \"value\" or \"percentage\"')
        total_factor = Portfolio['factor'].sum()
        Portfolio['w_factor_weight'] = Portfolio['factor']/total_factor
        del Portfolio['factor']

        return Portfolio


    # get stocks' daily data
    def get_stock_data(self):

        # can also use database
        stock_df = pd.DataFrame()
        f = open(self.Stock_dir,'rb')
        stock_data = pickle.load(f)
        # iterating all stocks and join them together
        for stock in stock_data:
            try:
                stock.set_index('tdate',inplace=True)
                stock['cum_ret'] = (stock.loc[self.end,'close']-stock.loc[self.start,'pclose'])/\
                    stock.loc[self.start,'pclose']
                stock = stock.loc[self.end]
            except:
                continue
            stock_df = stock_df.append(stock)
        # some id data is float. It should be integer
        stock_df['id'] = stock_df['id'].astype(int)
        stock_df = self.get_stock_code(stock_df)
        Stock_return = pd.DataFrame({'date':self.end,'id':stock_df['id'],\
            'cum_ret':stock_df['cum_ret'],'market_value':stock_df['market_value']})
        return Stock_return


    def get_merge_data_opt(self,Factor_dir):
        # get weights for three different kinds of portfolio
        Portfolio = self.get_portfolio()
        Portfolio_equal = self.get_portfolio_equal_weight(Factor_dir)
        Portfolio_factor = self.get_portfolio_factor_weight(Factor_dir)
        # merge the data for each portfolio
        merge_data = pd.merge(Portfolio,self.opt_merge_data,on='id',how='inner')
        merge_data = pd.merge(merge_data,self.stock,on='id',how='inner')
        merge_data_equal = pd.merge(Portfolio_equal,self.opt_merge_data,on='id',how='inner')
        merge_data_equal = pd.merge(merge_data_equal,self.stock,on='id',how='inner')
        merge_data_factor = pd.merge(Portfolio_factor,self.opt_merge_data,on='id',how='inner')
        merge_data_factor = pd.merge(merge_data_factor,self.stock,on='id',how='inner')

        return merge_data, merge_data_equal, merge_data_factor


    # get merge data for common portfolios
    def get_merge_data(self):

        merge_data_lst = []
        for dir in self.file_lst:
            Portfolio = pd.read_csv(dir)
            merge_data = pd.merge(Portfolio,self.opt_merge_data,on='id',how='inner')
            merge_data = pd.merge(merge_data,self.stock,on=['id','date'],how='inner')
            merge_data_lst.append(merge_data)

        return  merge_data_lst


    # get the weight of market value of stocks
    def get_market_value_weight(self,merge_data):

        market_value_weight = []
        for ind in self.Industry:
            mv = merge_data.loc[merge_data[ind]==1]
            if mv.empty:
                market_value_weight.append(0)
            else:
                market_value_weight.append(mv.groupby(['date',ind])[['market_value']].sum()['market_value'].iloc[0])
        # get market value weights for each stock
        market_value_weight /= np.sum(market_value_weight)
        industry_df = pd.DataFrame({'market_value_weight':market_value_weight})

        return industry_df


    # get the factor return data
    def get_factor_return(self,merge_data,weight_column):

        industry_df = self.get_market_value_weight(merge_data)
        Y = merge_data['cum_ret'] # cumulative return within a window
        X = merge_data[self.Industry+self.Style] # fatcor exposure data
        X = sm.add_constant(X) # add intercept

        weight = merge_data['market_value'].tolist()
        model = sm.WLS(Y,X,weights=weight)
        model_res = model.fit()
        # print(model_res.summary())
        # print(model_res.params)

        # use optimization to add constraints for WLS
        x = cp.Variable(len(self.Industry+self.Style))
        x.value = np.array(model_res.params[1:])
        A = np.array(merge_data[self.Industry+self.Style])
        y = np.array(merge_data['cum_ret'])
        objective = cp.Minimize(cp.sum_squares(y-A@x))
        # normalize the effect by industry
        constraints = [np.array(industry_df['market_value_weight'].tolist()+[0]*len(self.Style))@x==0]

        # define objective function for the problem and then solve it by mosek
        prob = cp.Problem(objective,constraints)
        prob.solve(cp.MOSEK)
        # get the weights of industry of each portfolio
        industry_weight, style_weight = [], []
        for factor in self.Industry:
            mv = merge_data.loc[merge_data[factor]==1]
            if mv.empty:
                industry_weight.append(0)
            else:
                temp = mv.groupby(['date',factor])[[weight_column]].sum()
                industry_weight.append(temp[weight_column].iloc[0])
        for factor in self.Style:
            style_weight.append(np.sum(merge_data[factor]*merge_data[weight_column]))

        return list(x.value), industry_weight, style_weight


    def get_portfolio_data_opt_pkl(self):

        FacRet_df = self.get_factor_name(pd.DataFrame({'factor':self.Industry+self.Style}))
        Industry_df = self.get_factor_name(pd.DataFrame({'factor':self.Industry}))

        factor_return, industry_weight, style_weight = self.get_factor_return(self.merge_data,'w_opt')
        # get the factor return of the portfolios
        FacRet_df['factor_return'], FacRet_df['portfolio_factor_return'] = factor_return, industry_weight+style_weight
        # get the weights of industry of portfolios
        Industry_df['industry_weight'] = industry_weight
        FacRet_df['return'] = FacRet_df['factor_return'] * FacRet_df['portfolio_factor_return']
        del FacRet_df['factor_return'], FacRet_df['portfolio_factor_return']

        factor_return, industry_weight, style_weight = self.get_factor_return(self.merge_data_equal,'w_equal_weight')
        FacRet_df['factor_return'], FacRet_df['portfolio_factor_return'] = factor_return, industry_weight+style_weight
        FacRet_df['return_equal'] = FacRet_df['factor_return'] * FacRet_df['portfolio_factor_return']
        Industry_df['industry_weight_equal'] = industry_weight
        del FacRet_df['factor_return'], FacRet_df['portfolio_factor_return']

        factor_return, industry_weight, style_weight = self.get_factor_return(self.merge_data_factor,'w_factor_weight')
        FacRet_df['factor_return'], FacRet_df['portfolio_factor_return'] = factor_return, industry_weight+style_weight
        FacRet_df['return_factor'] = FacRet_df['factor_return'] * FacRet_df['portfolio_factor_return']
        Industry_df['industry_weight_factor'] = industry_weight
        del FacRet_df['factor_return'], FacRet_df['portfolio_factor_return']

        # store the data in dictionary
        self.FacRet.update({self.end:FacRet_df})
        self.Industry_weight.update({self.end:Industry_df})

        # calculate the variance of the factor return
        industry = FacRet_df.iloc[:len(self.Industry)].copy()
        style = FacRet_df.iloc[len(self.Industry):].copy()
        FacVar = pd.DataFrame({'date':[self.end],'ret_ind':[np.var(industry['return'])],'ret_equ_ind':[np.var(industry['return_equal'])],
            'ret_fac_ind':[np.var(industry['return_factor'])],'ret_sty':[np.var(style['return'])],
            'ret_eql_sty':[np.var(style['return_equal'])],'ret_fac_sty':[np.var(style['return_factor'])]})
        self.FacVar.update({self.end:FacVar})

        # store the data
        self.Portfolio.update({self.end:pd.DataFrame({'date':self.end,\
            'id':self.merge_data['id'],'weight':self.merge_data['w_opt']})})
        self.Portfolio_equal.update({self.end:pd.DataFrame({'date':self.end,\
            'id':self.merge_data_equal['id'],'weight':self.merge_data_equal['w_equal_weight']})})
        self.Portfolio_factor.update({self.end:pd.DataFrame({'date':self.end,\
            'id':self.merge_data_factor['id'],'weight':self.merge_data_factor['w_factor_weight']})})


    # same process as the above function, but store data in .csv format
    def get_portfolio_data_opt_csv(self):

        self.makedir(self.Save_dir+r'\csv_data\%s_opt'%(self.window))
        FacRet_df = self.get_factor_name(pd.DataFrame({'factor':self.Industry+self.Style}))
        Industry_df = self.get_factor_name(pd.DataFrame({'factor':self.Industry}))

        factor_return, industry_weight, style_weight = self.get_factor_return(self.merge_data,'w_opt')
        FacRet_df['factor_return'], FacRet_df['portfolio_factor_return'] = factor_return, industry_weight+style_weight
        Industry_df['industry_weight'] = industry_weight
        FacRet_df['return'] = FacRet_df['factor_return'] * FacRet_df['portfolio_factor_return']
        del FacRet_df['factor_return'], FacRet_df['portfolio_factor_return']

        factor_return, industry_weight, style_weight = self.get_factor_return(self.merge_data_equal,'w_equal_weight')
        FacRet_df['factor_return'], FacRet_df['portfolio_factor_return'] = factor_return, industry_weight+style_weight
        FacRet_df['return_equal'] = FacRet_df['factor_return'] * FacRet_df['portfolio_factor_return']
        Industry_df['industry_weight_equal'] = industry_weight
        del FacRet_df['factor_return'], FacRet_df['portfolio_factor_return']

        factor_return, industry_weight, style_weight = self.get_factor_return(self.merge_data_factor,'w_factor_weight')
        FacRet_df['factor_return'], FacRet_df['portfolio_factor_return'] = factor_return, industry_weight+style_weight
        FacRet_df['return_factor'] = FacRet_df['factor_return'] * FacRet_df['portfolio_factor_return']
        Industry_df['industry_weight_factor'] = industry_weight
        del FacRet_df['factor_return'], FacRet_df['portfolio_factor_return']

        self.makedir(self.Save_dir+r'\csv_data\%s_opt\FacRet'%(self.window))
        FacRet_df.to_csv(self.Save_dir+r'\csv_data\%s_opt\FacRet\facret_%s_%s.csv'%(self.window,self.window,self.end),index=False)
        self.makedir(self.Save_dir+r'\csv_data\%s_opt\Industry_Weight'%(self.window))
        Industry_df.to_csv(self.Save_dir+r'\csv_data\%s_opt\Industry_Weight\industry_weight_%s_%s.csv'%(self.window,self.window,self.end),
            index=False)

        industry = FacRet_df.iloc[:len(self.Industry)].copy()
        style = FacRet_df.iloc[len(self.Industry):].copy()
        self.FacVar = {'date':[self.end],'ret_ind':[np.var(industry['return'])],'ret_equ_ind':[np.var(industry['return_equal'])],
            'ret_fac_ind':[np.var(industry['return_factor'])],'ret_sty':[np.var(style['return'])],
            'ret_eql_sty':[np.var(style['return_equal'])],'ret_fac_sty':[np.var(style['return_factor'])]}
        self.makedir(self.Save_dir+r'\csv_data\%s_opt\FacVar'%(self.window))
        pd.DataFrame(self.FacVar).to_csv(self.Save_dir+r'\csv_data\%s_opt\FacVar\facvar_%s_%s.csv'%(self.window,self.window,self.end),index=False)

        self.makedir(self.Save_dir+r'\csv_data\%s_opt\Portfolio'%(self.window))
        pd.DataFrame({'date':self.end,\
            'id':self.merge_data['id'],'weight':self.merge_data['w_opt']})\
            .to_csv(self.Save_dir+r'\csv_data\%s_opt\Portfolio\portfolio_%s_%s.csv'%(self.window,self.window,self.end),index=False)

        self.makedir(self.Save_dir+r'\csv_data\%s_opt\Portfolio_Equal'%(self.window))
        pd.DataFrame({'date':self.end,\
            'id':self.merge_data_equal['id'],'weight':self.merge_data_equal['w_equal_weight']})\
            .to_csv(self.Save_dir+r'\csv_data\%s_opt\Portfolio_Equal\portfolio_equal_%s_%s.csv'%(self.window,self.window,self.end),index=False)

        self.makedir(self.Save_dir+r'\csv_data\%s_opt\Portfolio_Factor'%(self.window))
        pd.DataFrame({'date':self.end,\
            'id':self.merge_data_factor['id'],'weight':self.merge_data_factor['w_factor_weight']})\
            .to_csv(self.Save_dir+r'\csv_data\%s_opt\Portfolio_Factor\portfolio_factor_%s_%s.csv'%(self.window,self.window,self.end),index=False)


    # same process as the above function, but no opimization needed
    def get_portfolio_data_pkl(self):

        FacRet_df = self.get_factor_name(pd.DataFrame({'factor':self.Industry+self.Style}))
        Industry_df = self.get_factor_name(pd.DataFrame({'factor':self.Industry}))

        for i in range(len(self.merge_data_lst)):
            factor_return, industry_weight, style_weight = self.get_factor_return(self.merge_data_lst[i],'weight')
            FacRet_df['factor_return'], FacRet_df['portfolio_factor_return'] = factor_return, industry_weight+style_weight
            Industry_df['industry_weight_%s'%(i+1)] = industry_weight
            FacRet_df['return_%s'%(i+1)] = FacRet_df['factor_return'] * FacRet_df['portfolio_factor_return']
            del FacRet_df['factor_return'], FacRet_df['portfolio_factor_return']

        self.FacRet.update({self.end:FacRet_df})
        self.Industry_weight.update({self.end:Industry_df})
        industry = FacRet_df.iloc[:len(self.Industry)].copy()
        style = FacRet_df.iloc[len(self.Industry):].copy()

        FacVar = {'date':[self.end]}
        for i in range(len(self.merge_data_lst)):
            FacVar.update({'return_industry_%s'%(i+1):np.var(industry['return_%s'%(i+1)])})
            FacVar.update({'return_style_%s'%(i+1):np.var(style['return_%s'%(i+1)])})
            self.Portfolio_lst[i].update({self.end:pd.DataFrame({'date':self.end,\
                'id':self.merge_data_lst[i]['id'],'weight':self.merge_data_lst[i]['weight']})})

        FacVar = pd.DataFrame(FacVar)
        self.FacVar.update({self.end:FacVar})


    # same process as the above function, but no opimization needed and stored in .csv format
    def get_portfolio_data_csv(self):

        self.makedir(self.Save_dir+r'\csv_data\%s'%(self.window))
        FacRet_df = self.get_factor_name(pd.DataFrame({'factor':self.Industry+self.Style}))
        Industry_df = self.get_factor_name(pd.DataFrame({'factor':self.Industry}))

        for i in range(len(self.merge_data_lst)):
            factor_return, industry_weight, style_weight = self.get_factor_return(self.merge_data_lst[i],'weight')
            FacRet_df['factor_return'], FacRet_df['portfolio_factor_return'] = factor_return, industry_weight+style_weight
            Industry_df['industry_weight_%s'%(i+1)] = industry_weight
            FacRet_df['return_%s'%(i+1)] = FacRet_df['factor_return'] * FacRet_df['portfolio_factor_return']
            del FacRet_df['factor_return'], FacRet_df['portfolio_factor_return']

        self.makedir(self.Save_dir+r'\csv_data\%s\FacRet'%(self.window))
        FacRet_df.to_csv(self.Save_dir+r'\csv_data\%s\FacRet\facret_%s_%s.csv'%(self.window,self.window,self.end),index=False)
        self.makedir(self.Save_dir+r'\csv_data\%s\Industry_Weight'%(self.window))
        Industry_df.to_csv(self.Save_dir+r'\csv_data\%s\Industry_Weight\industry_weight_%s_%s.csv'%(self.window,self.window,self.end),
            index=False)
        industry = FacRet_df.iloc[:len(self.Industry)].copy()
        style = FacRet_df.iloc[len(self.Industry):].copy()

        FacVar = {'date':[self.end]}
        for i in range(len(self.merge_data_lst)):
            FacVar.update({'return_industry_%s'%(i+1):[np.var(industry['return_%s'%(i+1)])]})
            FacVar.update({'return_style_%s'%(i+1):[np.var(style['return_%s'%(i+1)])]})
            self.makedir(self.Save_dir+r'\csv_data\%s\Portfolio_%s'%(self.window,i+1))
            pd.DataFrame({'date':self.end,'id':self.merge_data_lst[i]['id'],\
                'weight':self.merge_data_lst[i]['weight']}).to_csv(self.Save_dir+r'\csv_data\%s\Portfolio_%s'%(self.window,i+1)+\
                    r'\portfolio_%s_%s.csv'%(self.window,self.end),index=False)

        self.makedir(self.Save_dir+r'\csv_data\%s\FacVar'%(self.window))
        pd.DataFrame(self.FacVar).to_csv(self.Save_dir+r'\csv_data\%s\FacVar\facvar_%s_%s.csv'%(self.window,self.window,self.end),index=False)


    # store all the data
    def store_data(self):
        print('start processing data')
        self.Time_data = self.get_time_series()
        start_time = str(self.Time_data[0])
        end_time = str(self.Time_data[-1])
        Bench_lst = self.get_fileLst(self.Bench_dir,start_time,end_time)
        factor_lst = self.get_fileLst(self.Factor_dir,start_time,end_time)
        # do the optimization, then store the data of optimization portfolio, equal weighted portfolio, and factor weighted portfolio
        if self.optimization:
            print('now getting required data for optimal portfolio')
            for i in range(len(self.Time_data)-self.window):
                self.start = self.Time_data[i]
                self.end = self.Time_data[i+self.window]
                # the format of the saved data
                if self.format == 'csv':
                    if os.path.exists(self.Save_dir+r'\csv_data\%s_opt\Portfolio\portfolio_%s_%s.csv'%(self.window,self.window,self.end)):
                        continue
                if self.format == 'pkl':
                    if os.path.exists(self.Save_dir+r'\pkl_data\Portfolio_%s_%s_%s.pkl'%(self.window,self.Time_data[self.window],self.end_date)):
                        continue
                # determine path based on the current trading date
                Bench_dir = Bench_lst[i+self.window]
                Factor_dir = factor_lst[i+self.window]
                # get data used for optimization
                self.opt_merge_data = self.get_opt_merge_data(Bench_dir,Factor_dir)
                self.stock = self.get_stock_data()
                self.merge_data, self.merge_data_equal,self.merge_data_factor = self.get_merge_data_opt(Factor_dir)
                # change the current date to the next date, get ready for processing the next trading date's data
                self.current = self.end
                if self.format == 'pkl':
                    self.get_portfolio_data_opt_pkl()
                elif self.format == 'csv':
                    self.get_portfolio_data_opt_csv()
                else:
                    raise ValueError(u'The value of "self.format" should be "pkl" or "csv"')
                print('optimal portfolio data from %s to %s has been updated'%(self.start,self.end))
        # no optimization, just input the weights of stock of portfolios
        else:
            print('now getting required data for common portfolio')
            for i in self.portfolio_lst:
                self.portfolio_lst_lst.append(self.get_fileLst(i,start_time,end_time))
            for i in range(len(self.Time_data)-self.window):
                # initilize the date data for each date
                self.start = self.Time_data[i]
                self.end = self.Time_data[i+self.window]
                # if required data already exists, just skip it
                if self.format == 'pkl':
                    if os.path.exists(self.Save_dir+r'\pkl_data\Portfolio1_%s_%s_%s.pkl'%(self.window,self.Time_data[self.window],self.end_date)):
                        break
                elif self.format == 'csv':
                    if os.path.exists(self.Save_dir+r'\csv_data\%s\Portfolio1\portfolio_1_%s.csv'%(self.window,self.end)):
                        continue
                # change the path
                Bench_dir = Bench_lst[i+self.window]
                Factor_dir = factor_lst[i+self.window]
                self.change_portfolio_dir(i+self.window)
                # precess the data
                self.opt_merge_data = self.get_opt_merge_data(Bench_dir,Factor_dir)
                self.stock = self.get_stock_data()
                self.merge_data_lst = self.get_merge_data()
                # change the current date to the next date, get ready for processing the next trading date's data
                self.current = self.end
                # choose the format of stored data
                if self.format == 'pkl':
                    self.get_portfolio_data_pkl()
                elif self.format == 'csv':
                    self.get_portfolio_data_csv()
                else:
                    raise ValueError(u'The value of "self.format" should be "pkl" or "csv"')
                print('optimal portfolio data from %s to %s has been updated'%(self.start,self.end))
        # if the format is .pkl, we should do one step further to store the data
        if self.format == 'pkl':
            self.makedir(self.Save_dir+r'\pkl_data')
            if self.optimization:
                path_lst = [self.Save_dir+r'\pkl_data\FacRet_%s_%s_%s_opt.pkl'%(self.window,self.Time_data[self.window],self.end_date),
                    self.Save_dir+r'\pkl_data\Industry_Weight_%s_%s_%s_opt.pkl'%(self.window,self.Time_data[self.window],self.end_date),
                    self.Save_dir+r'\pkl_data\FacVar_%s_%s_%s_opt.pkl'%(self.window,self.Time_data[self.window],self.end_date),
                    self.Save_dir+r'\pkl_data\Portfolio_%s_%s_%s.pkl'%(self.window,self.Time_data[self.window],self.end_date),
                    self.Save_dir+r'\pkl_data\Portfolio_equal_%s_%s_%s.pkl'%(self.window,self.Time_data[self.window],self.end_date),
                    self.Save_dir+r'\pkl_data\Portfolio_factor_%s_%s_%s.pkl'%(self.window,self.Time_data[self.window],self.end_date)]
                dict_lst = [self.FacRet,self.Industry_weight,self.FacVar,self.Portfolio,self.Portfolio_equal,self.Portfolio_factor]
            else:
                path_lst = [self.Save_dir+r'\pkl_data\FacRet_%s_%s_%s.pkl'%(self.window,self.Time_data[self.window],self.end_date),
                    self.Save_dir+r'\pkl_data\Industry_Weight_%s_%s_%s.pkl'%(self.window,self.Time_data[self.window],self.end_date),
                    self.Save_dir+r'\pkl_data\FacVar_%s_%s_%s.pkl'%(self.window,self.Time_data[self.window],self.end_date)]\
                    +[self.Save_dir+r'\pkl_data\Portfolio_%s'%(i+1)+r'_%s_%s_%s.pkl'%(self.window,self.Time_data[self.window],self.end_date)
                        for i in range(len(self.merge_data_lst))]
                dict_lst = [self.FacRet,self.Industry_weight,self.FacVar]+[self.Portfolio_lst[i] for i in
                    range(len(self.merge_data_lst))]
            self.save_pkl(path_lst,dict_lst)


# if __name__ == "__main__":
#     start = time.time()
#     data = getData()
#     data.store_data()
#     end = time.time()
#     print('Total time used: '+str(end-start)) # about 45 seconds to get data of one trading day