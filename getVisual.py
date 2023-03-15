import os
import pickle
import numpy as np
import pandas as pd
from VisualInfo import VisualInfo


class getVisual(VisualInfo):
    def __init__(self) -> None:
        VisualInfo.__init__(self)
        self.Time_data = None # trading date series from the start trading day to the end trading day
        self.stock = None # individual stock data
        self.PB_PE = None # PB、PE data
        self.ROE = None # ROE data
        self.merge_data_lst = []
        self.csvdir_lst = [i for i in [self.Portfolio1Csv_dir,self.Portfolio2Csv_dir,self.Portfolio3Csv_dir,
            self.Portfolio4Csv_dir,self.Portfolio5Csv_dir,self.Portfolio6Csv_dir,self.Portfolio7Csv_dir,
            self.Portfolio8Csv_dir,self.Portfolio9Csv_dir,self.Portfolio10Csv_dir] if i is not None]
        self.pkldir_lst = [i for i in [self.Portfolio1Pkl_dir,self.Portfolio2Pkl_dir,self.Portfolio3Pkl_dir,
            self.Portfolio4Pkl_dir,self.Portfolio5Pkl_dir,self.Portfolio6Pkl_dir,self.Portfolio7Pkl_dir,
            self.Portfolio8Pkl_dir,self.Portfolio9Pkl_dir,self.Portfolio10Pkl_dir] if i is not None]
        self.csvdir_lst_lst = []
        if self.format == 'csv':
            self.file_lst = [0]*len(self.csvdir_lst)
        elif self.format == 'pkl':
            self.file_lst = [0]*len(self.pkldir_lst)


    # turn the format of one column into xxxxxx.SZ
    def get_stock_code(self,df,column='id'):
        df[column] = df[column].astype(str)
        df[column] = df[column].apply(lambda x: x.zfill(6))
        df[column] = df[column].apply(lambda x: x + '.SH' if x[0] == '6' else x + '.SZ')
        return df


    # turn the format of one column into Timestamp
    def get_date_df(self,df,column='date'):
        df[column] = df[column].apply(lambda x: pd.Timestamp(int(str(x)[:4]),int(str(x)[4:6]),int(str(x)[6:])))
        return df


    # change the format of date
    def date2quarter(self):
        end_string = str(self.current)
        if int(end_string[4:6]) <= 3:
            return int(end_string[:4]+'0331')
        elif int(end_string[4:6]) <= 6:
            return int(end_string[:4]+'0630')
        elif int(end_string[4:6]) <= 3:
            return int(end_string[:4]+'0930')
        else:
            return int(str(int(end_string[:4])-1)+'1231')


    # use weighted average to fill the NA. "column" is the column to fill. "weight" is the column with weight.
    def fill_null_value(self,df,column,weight):
        return df[column].fillna(np.average(df[column].dropna().tolist(),weights=df[[weight,column]].dropna()[weight].tolist()))

    # get the time series data
    def get_time_series(self):

        # use database
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
                return [Time_lst[start_index]]

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
            
        return Time_lst[start_index:end_index]


    # get the list containing the required files
    def get_fileLst(self,path,start,end):
        
        path_lst = sorted(os.listdir(path))
        start_path = [file for file in path_lst if start in file][0]
        end_path = [file for file in path_lst if end in file][0]
        start_index = path_lst.index(start_path)
        end_index = path_lst.index(end_path)

        return path_lst[start_index:end_index+1]


    # change the path to read different data files
    def change_portfolio_dir(self,n):
        for i in range(len(self.csvdir_lst_lst)):
            self.file_lst[i] = r'%s\%s'%(self.csvdir_lst[i],self.csvdir_lst_lst[i][n])


    # get the stocks' daily data
    def get_stock_data(self):
        # use database
        stock_df = pd.DataFrame()
        f = open(self.Stock_dir,'rb')
        stock_data = pickle.load(f)
        for stock in stock_data:
            try:
                stock.set_index('tdate',inplace=True)
                stock['bench_ret'] = stock.loc[self.current,'zz500_PCT_CHG']/100
                stock = stock.loc[self.current]
            except:
                continue
            stock_df = stock_df.append(stock)
        stock_df['id'] = stock_df['id'].astype(int)
        stock_df = self.get_stock_code(stock_df)
        self.stock = pd.DataFrame({'date':self.current,'id':stock_df['id'],'bench':stock_df['zz500_PRE_CLOSE'],
        'bench_ret':stock_df['bench_ret'],'pclose':stock_df['pclose'],'close':stock_df['close']})


    # get PB, PE, and ROE data
    def get_PB_PE_ROE(self,PB_PE_dir):
        # use database
        PB_PE_df = pd.read_csv(r'%s\%s'%(self.PBPE_dir,PB_PE_dir))[['S_INFO_WINDCODE',self.PB_col,self.PE_col,'TRADE_DT']]
        ROE_df = pd.read_csv(self.ROE_dir)[['S_INFO_WINDCODE',self.ROE_col,'REPORT_PERIOD']]
        end_date = self.date2quarter()
        ROE_df = ROE_df.loc[ROE_df['REPORT_PERIOD']<=end_date]
        ROE_df = ROE_df.sort_values(by=['REPORT_PERIOD'])
        ROE_df = ROE_df.drop_duplicates(subset=['S_INFO_WINDCODE'],keep='last')
        self.PB_PE = PB_PE_df.rename(columns={'S_INFO_WINDCODE':'id','TRADE_DT':'date'})
        self.ROE = ROE_df.rename(columns={'S_INFO_WINDCODE':'id'})


    def merge(self,origin_data,data_lst,on=['id','date'],how='left'):
        for df in data_lst:
            origin_data = pd.merge(origin_data,df,on=on,how=how)
        return origin_data


    # merge the csv data
    def get_merge_data_csv(self,data_lst):
        self.merge_data_lst = []
        for i in range(len(self.file_lst)):
            try:
                Portfolio = pd.read_csv(self.file_lst[i])
                Portfolio.rename(columns={'weight':'w_%s'%(i+1)},inplace=True)
                Portfolio = self.merge(Portfolio,data_lst)
                Portfolio = pd.merge(Portfolio,self.ROE,on='id',how='left')
                self.merge_data_lst.append(Portfolio)
            except:
                continue


    # merge the pkl data
    def get_merge_data_pkl(self,data_lst):
        self.merge_data_lst = []
        for i in range(len(self.pkldir_lst)):
            try:
                f = open(self.pkldir_lst[i],'rb')
                Portfolio = pickle.load(f)[self.current]
                Portfolio.rename(columns={'weight':'w_%s'%(i+1)},inplace=True)
                Portfolio = self.merge(Portfolio,data_lst)
                self.merge_data_lst.append(Portfolio)
            except:
                continue


    def update_data_csv(self,FacRetCsv_dir,FacVarCsv_dir,IndustryWeightCsv_dir):

        # get factor return data
        FacRet = pd.read_csv(r'.\%s\%s'%(self.FacRetCsv_dir,FacRetCsv_dir))
        FacRet.columns = ['factor']+['ret_%s'%(i+1) for i in range(self.NumPortfolio)]

        # get variance of factor return
        FacVar = pd.read_csv(r'.\%s\%s'%(self.FacVarCsv_dir,FacVarCsv_dir))
        FacVar.columns = ['date']+['var_%s_ind'%(i+1) for i in range(self.NumPortfolio)]\
            +['var_%s_sty'%(i+1) for i in range(self.NumPortfolio)]
        FacVar['date'] = FacVar['date'].astype(int)

        # get industry weight data
        Industry_Weight = pd.read_csv(r'.\%s\%s'%(self.IndustryWeightCsv_dir,IndustryWeightCsv_dir))
        Industry_Weight.columns = ['factor']+['w_%s'%(i+1) for i in range(self.NumPortfolio)]
        self.FacRet_dict.update({self.current:FacRet})
        self.FacVar_dict.update({self.current:FacVar})
        self.IndustryWeight_dict.update({self.current:Industry_Weight})

        # get stocks' weight and close price for each portfolio
        self.Stock_dict.update({self.current:[self.merge_data_lst[i][['w_%s'%(i+1),'pclose']]\
            for i in range(self.NumPortfolio)]})

        # benchmark return
        bench_ret = self.merge_data_lst[0]['bench_ret'].iloc[0]
        bench = self.merge_data_lst[0]['bench'].iloc[0]

        if not self.Info_dict.get('date'):
            self.Info_dict.update({'date':[self.current],'bench':[bench],'bench_ret':[bench_ret]})
        else:
            self.Info_dict['date'].append(self.current)
            self.Info_dict['bench'].append(bench)
            self.Info_dict['bench_ret'].append(bench_ret)
        for i in range(self.NumPortfolio):
            cum_ret = ((self.merge_data_lst[i]['close']*self.merge_data_lst[i]['w_%s'%(i+1)]).sum()\
                -(self.merge_data_lst[i]['pclose']*self.merge_data_lst[i]['w_%s'%(i+1)]).sum())\
                /(self.merge_data_lst[i]['pclose']*self.merge_data_lst[i]['w_%s'%(i+1)]).sum()

        # fill the NA of PB, PE, and ROE data
            self.merge_data_lst[i][self.PB_col] = self.fill_null_value(self.merge_data_lst[i],self.PB_col,'w_%s'%(i+1))
            self.merge_data_lst[i][self.PE_col] = self.fill_null_value(self.merge_data_lst[i],self.PE_col,'w_%s'%(i+1))
            self.merge_data_lst[i][self.ROE_col] = self.fill_null_value(self.merge_data_lst[i],self.ROE_col,'w_%s'%(i+1))

        # get PB、PE、and ROE data of the portfolio
            cum_PB = (((self.merge_data_lst[i][self.PB_col]*self.merge_data_lst[i]['w_%s'%(i+1)]).sum()))
            cum_PE = (((self.merge_data_lst[i][self.PE_col]*self.merge_data_lst[i]['w_%s'%(i+1)]).sum()))
            cum_ROE = (((self.merge_data_lst[i][self.ROE_col]*self.merge_data_lst[i]['w_%s'%(i+1)]).sum()))

        # store the data
            if not self.Info_dict.get('ret_%s'%(i+1)):
                self.Info_dict.update({'ret_%s'%(i+1):[cum_ret],'PB_%s'%(i+1):[cum_PB],
                    'PE_%s'%(i+1):[cum_PE],'ROE_%s'%(i+1):[cum_ROE]})
            else:
                self.Info_dict['ret_%s'%(i+1)].append(cum_ret)
                self.Info_dict['PB_%s'%(i+1)].append(cum_PB)
                self.Info_dict['PE_%s'%(i+1)].append(cum_PE)
                self.Info_dict['ROE_%s'%(i+1)].append(cum_ROE)


    # same process as the above function, but read the pkl data
    def update_data_pkl(self):

        f = open(self.FacRetPkl_dir,'rb')

        FacRet = pickle.load(f)[self.current]
        FacRet.columns = ['factor']+['ret_%s'%(i+1) for i in range(self.NumPortfolio)]

        f = open(self.FacVarPkl_dir,'rb')
        FacVar = pickle.load(f)[self.current]
        FacVar.columns = ['date']+['var_%s_ind'%(i+1) for i in range(self.NumPortfolio)]\
            +['var_%s_sty'%(i+1) for i in range(self.NumPortfolio)]
        FacVar['date'] = FacVar['date'].astype(int)

        f = open(self.IndustryWeightPkl_dir,'rb')
        Industry_Weight = pickle.load(f)[self.current]
        Industry_Weight.columns = ['factor']+['w_%s'%(i+1) for i in range(self.NumPortfolio)]
        self.FacRet_dict.update({self.current:FacRet})
        self.FacVar_dict.update({self.current:FacVar})
        self.IndustryWeight_dict.update({self.current:Industry_Weight})

        self.Stock_dict.update({self.current:[self.merge_data_lst[i][['w_%s'%(i+1),'pclose']]\
            for i in range(self.NumPortfolio)]})

        bench_ret = self.merge_data_lst[0]['bench_ret'].iloc[0]
        bench = self.merge_data_lst[0]['bench'].iloc[0]

        if not self.Info_dict.get('date'):
            self.Info_dict.update({'date':[self.current],'bench':[bench],'bench_ret':[bench_ret]})
        else:
            self.Info_dict['date'].append(self.current)
            self.Info_dict['bench'].append(bench)
            self.Info_dict['bench_ret'].append(bench_ret)
        for i in range(self.NumPortfolio):
            cum_ret = ((self.merge_data_lst[i]['close']*self.merge_data_lst[i]['w_%s'%(i+1)]).sum()\
                -(self.merge_data_lst[i]['pclose']*self.merge_data_lst[i]['w_%s'%(i+1)]).sum())\
                /(self.merge_data_lst[i]['pclose']*self.merge_data_lst[i]['w_%s'%(i+1)]).sum()

            self.merge_data_lst[i][self.PB_col] = self.fill_null_value(self.merge_data_lst[i],self.PB_col,'w_%s'%(i+1))
            self.merge_data_lst[i][self.PE_col] = self.fill_null_value(self.merge_data_lst[i],self.PE_col,'w_%s'%(i+1))
            self.merge_data_lst[i][self.ROE_col] = self.fill_null_value(self.merge_data_lst[i],self.ROE_col,'w_%s'%(i+1))

            cum_PB = (((self.merge_data_lst[i][self.PB_col]*self.merge_data_lst[i]['w_%s'%(i+1)]).sum()))
            cum_PE = (((self.merge_data_lst[i][self.PE_col]*self.merge_data_lst[i]['w_%s'%(i+1)]).sum()))
            cum_ROE = (((self.merge_data_lst[i][self.ROE_col]*self.merge_data_lst[i]['w_%s'%(i+1)]).sum()))

            if not self.Info_dict.get('ret_%s'%(i+1)):
                self.Info_dict.update({'ret_%s'%(i+1):[cum_ret],'PB_%s'%(i+1):[cum_PB],
                    'PE_%s'%(i+1):[cum_PE],'ROE_%s'%(i+1):[cum_ROE]})
            else:
                self.Info_dict['ret_%s'%(i+1)].append(cum_ret)
                self.Info_dict['PB_%s'%(i+1)].append(cum_PB)
                self.Info_dict['PE_%s'%(i+1)].append(cum_PE)
                self.Info_dict['ROE_%s'%(i+1)].append(cum_ROE)

    # get all the data used for visualize
    def get_data(self):

        print('Start processing the data')
        self.Time_data = self.get_time_series()
        start_time = str(self.Time_data[0])
        end_time = str(self.Time_data[-1])
        PB_PE_lst = self.get_fileLst(self.PBPE_dir,start_time,end_time)
        if self.format == 'csv':
            FacRetCsv_lst = self.get_fileLst(self.FacRetCsv_dir,start_time,end_time)
            FacVarCsv_lst = self.get_fileLst(self.FacVarCsv_dir,start_time,end_time)
            IndustryWeightCsv_lst = self.get_fileLst(self.IndustryWeightCsv_dir,start_time,end_time)
            for i in self.csvdir_lst:
                self.csvdir_lst_lst.append(self.get_fileLst(i,start_time,end_time))
            for i in range(len(self.Time_data)):
                if not self.FacRet_dict.get(self.Time_data[i]):
                    FacRetCsv_dir = FacRetCsv_lst[i]
                    FacVarCsv_dir = FacVarCsv_lst[i]
                    IndustryWeightCsv_dir = IndustryWeightCsv_lst[i]
                    self.current = self.Time_data[i]
                    self.get_stock_data()
                    self.get_PB_PE_ROE(PB_PE_lst[i])
                    data_lst = [self.stock,self.PB_PE]
                    self.change_portfolio_dir(i)
                    self.get_merge_data_csv(data_lst)
                    self.update_data_csv(FacRetCsv_dir,FacVarCsv_dir,IndustryWeightCsv_dir)
                print('Data from %s has been uploaded'%(self.Time_data[i]))
        elif self.format == 'pkl':
            for i in range(len(self.Time_data)):
                if not self.FacRet_dict.get(self.Time_data[i]):
                    self.current = self.Time_data[i]
                    self.get_stock_data()
                    self.get_PB_PE_ROE(PB_PE_lst[i])
                    data_lst = [self.stock,self.PB_PE]
                    self.get_merge_data_pkl(data_lst)
                    self.update_data_pkl()
                print('Data from %s has been uploaded'%(self.Time_data[i]))
        self.Info_df = pd.DataFrame(self.Info_dict)
        self.Info_df['time'] = self.Info_df['date']
        self.Info_df.set_index('time',inplace=True)
        self.current = self.start_date

        return self.Time_data,self.Stock_dict,self.FacRet_dict,self.FacVar_dict,self.IndustryWeight_dict,self.Info_df