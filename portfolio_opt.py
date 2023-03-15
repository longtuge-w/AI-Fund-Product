import os
import pickle
import pandas as pd
import numpy as np
import cvxpy as cp
import scipy.io as scio
import matlab.engine
from scipy.linalg import sqrtm
from cvxpy.atoms.norm import norm
from DataInfo import DataInfo


class portfolio_opt(DataInfo):
    def __init__(self):
        DataInfo.__init__(self)
        self.current = None # current date


    # turn the format of one column into xxxxxx.SZ
    def get_stock_code(self,df,column='id'):
        df[column] = df[column].astype(str)
        df[column] = df[column].apply(lambda x: x.zfill(6))
        df[column] = df[column].apply(lambda x: x + '.SH' if x[0] == '6' else x + '.SZ')
        return df


    # create files
    def makedir(self,path):
        folder = os.path.exists(path)
        if not folder:
            os.makedirs(path)


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


    # get the covariance data of factor
    def get_covariance(self,Covariance_dir):
        # can also use database
        cov_text = pd.read_csv(r'%s\%s'%(self.Covariance_dir,Covariance_dir),skiprows=[0,1],sep='|')
        # reset the index
        cov_text.index = range(cov_text.shape[0])
        # initialize the covariance data
        cov_data = pd.DataFrame(np.zeros(len(self.Industry+self.Style)**2).\
            reshape(len(self.Industry+self.Style),len(self.Industry+self.Style)),\
            index=self.Industry+self.Style,columns=self.Industry+self.Style)
        # get the whole covariance data
        for i in range(cov_text.shape[0]-1):
            if cov_text.iloc[i,0] in self.Industry+self.Style and cov_text.iloc[i,1] in self.Industry+self.Style:
                cov_data[cov_text.iloc[i,0]][cov_text.iloc[i,1]] = cov_text.iloc[i,2]/100
                cov_data[cov_text.iloc[i,1]][cov_text.iloc[i,0]] = cov_text.iloc[i,2]/100
        return cov_data


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


    # optimization by python
    def opt_python(self):
        FacScore = np.array(self.opt_merge_data['factor'])
        BenWeight = np.array(self.opt_merge_data['i_weight'])
        LastWeight = np.array(self.opt_merge_data['w_last'])
        X_Industry = np.array(self.opt_merge_data[self.Industry])
        X_Style = np.array(self.opt_merge_data[self.Style])
        FacCov = self.cov_data.values
        X_cov = np.array(self.opt_merge_data[self.Industry+self.Style])

        n = FacScore.shape[0]
        StockCov = np.array(X_cov @ FacCov @ X_cov.T)
        V, D = np.linalg.eig(StockCov)
        M = sqrtm(D) @ V.T

        weight = cp.Variable(n)
        objective = FacScore @ weight
        track = norm(M @ (weight-BenWeight))
        turnover = np.ones(n).T @ cp.abs(weight-LastWeight)
        FacStyle = X_Style.T @ (weight-BenWeight)
        FacInd = X_Industry.T @ (weight-BenWeight)

        contraints = [cp.sum(turnover) <= 1.1, # turnover constraint
            cp.sum(weight) == 1,
            weight >= 0, weight <= 0.02, # constraint the weight of individual stock
            FacStyle >= -0.01, FacStyle <= 0.01, # constraint the style factor
            FacInd >= -0.01, FacInd <= 0.01, # constraint the industry factor
            track <= 0.5/(48**0.5) # constraint the tracking error
        ]
        # define the objective function, then solve it by optimizer mosek
        prob = cp.Problem(cp.Maximize(objective), constraints=contraints)
        prob.solve(solver=cp.MOSEK)
        weight_df = pd.DataFrame(list(weight.value), columns=['w_opt'])
        # save the result of portfolio optimization
        self.makedir(self.Save_dir+r'.\cvx_portfolio')
        weight_df.to_csv(self.Save_dir+r'\cvx_portfolio\%sportfolio.csv' % (self.end),index=False)
        print('The optimal portfolio on %s is updated!'%(self.end))


    # get data in .mat format
    def get_mat_data(self):
        covariance = self.cov_data
        scio.savemat(self.Save_dir+r'\data.mat',{'MU':np.array(self.opt_merge_data['factor']),'bmw':np.array(self.opt_merge_data['i_weight']),
            'W0':np.array(self.opt_merge_data['w_last']),'Ind':np.array(self.opt_merge_data[self.Industry]),
            'Style':np.array(self.opt_merge_data[self.Style]),'Index':np.array(self.opt_merge_data['id']),
            'COV':covariance.values,'X_cov':np.array(self.opt_merge_data[self.Industry+self.Style])})


    # optimization by Matlab
    def opt_matlab(self):
        self.makedir(self.Save_dir+r'.\cvx_portfolio')
        engine = matlab.engine.start_matlab()
        data = scio.loadmat(self.Save_dir+r'\data.mat')
        info = engine.Opt(self.Cvx_dir,self.Save_dir+r'\cvx_portfolio\%sportfolio.csv' % (self.end),
            matlab.double(data['MU'].tolist()),matlab.double(data['COV'].tolist()),matlab.double(data['X_cov'].tolist()),
            matlab.double(data['Ind'].tolist()),matlab.double(data['Style'].tolist()),matlab.double(data['bmw'].tolist()),
            matlab.double(data['W0'].tolist()))
        print(info)


    # store all the data
    def opt(self):
        print('Start processing the data')
        self.Time_data = self.get_time_series()
        start_time = str(self.Time_data[0])
        end_time = str(self.Time_data[-1])
        Bench_lst = self.get_fileLst(self.Bench_dir,start_time,end_time)
        factor_lst = self.get_fileLst(self.Factor_dir,start_time,end_time)
        covariance_lst = self.get_fileLst(self.Covariance_dir,start_time,end_time)
        # do the optimization, then store the data of optimization portfolio, equal weighted portfolio, and factor weighted portfolio
        if self.optimization:
            for i in range(len(self.Time_data)-self.window):
                self.start = self.Time_data[i]
                self.end = self.Time_data[i+self.window]
                # the format of the saved data
                if os.path.exists(self.Save_dir+r'\cvx_portfolio\%sportfolio.csv' % (self.end)):
                    continue
                # determine path based on the current trading date
                Bench_dir = Bench_lst[i+self.window]
                Factor_dir = factor_lst[i+self.window]
                Covariance_dir = covariance_lst[i+self.window]
                # get data used for optimization
                self.cov_data = self.get_covariance(Covariance_dir)
                self.opt_merge_data = self.get_opt_merge_data(Bench_dir,Factor_dir)
                # choose Matlab or python to do the optimization
                if self.OptCode == 'matlab':
                    self.get_mat_data()
                    self.opt_matlab()
                elif self.OptCode == 'python':
                    self.opt_python()
                self.current = self.end
                print('optimal portfolio data from %s to %s has been updated'%(self.start,self.end))
        print('portfolio optimization finished')