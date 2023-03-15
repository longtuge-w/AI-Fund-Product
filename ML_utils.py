import os
import sys
import pickle
from tempfile import tempdir
from time import time
from git import DiffIndex
import pandas as pd
import numpy as np
import warnings
from tqdm import trange
import joblib
import jieba
import jieba.posseg
import jieba.analyse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score


# used to ignore the warnings
warnings.filterwarnings("ignore")

# number of days for each month
month_to_days = {
    1: 31, 2: 28, 3: 31, 4: 30,
    5: 30, 6: 31, 7: 31, 8: 31,
    9: 30, 10: 31, 11: 30, 0: 31,
}

def read_tradeDate_data(time_dir: str, start_date: int, end_date: int):
    # read the data for all trading dates
    f = open(time_dir,'rb')
    timeLst = pickle.load(f)
    timeLst = timeLst[(timeLst>=start_date) & (timeLst<=end_date)]
    f.close()
    return timeLst


def read_date_data(notice_dir: str, start_date: int, end_date: int):
    # read the data for all dates
    timeLst = os.listdir(notice_dir)
    timeLst = np.array(sorted([int(i[-13:-5]) for i in timeLst]))
    timeLst = timeLst[(timeLst >= start_date) & (timeLst <= end_date)]
    return timeLst


def read_stock_data(stocklist_dir: str, start_date: int, end_date: int, features: list, save_dir: str, stockPool: str='zz500',
    benchmark: str='zz500', benchPerc: float=0.5, back_window: int=1, forward_window: int=1, ):

    # create folder to store data
    if not os.path.exists(r'%s\\Stock_Data' % save_dir):
        os.makedirs(r'%s\\Stock_Data' % save_dir)
    
    if os.path.exists(r'%s\\Stock_Data\\Stock_Data_%s_%s_%s_%s.pkl' % (save_dir, start_date, end_date, back_window, forward_window)):
        print('Required stock data already exists')
        return None

    # choose benchmark to calculate the abnormal return
    if benchmark == 'hs300':
        bench = 'hs300_CLOSE'
    elif benchmark == 'zz500':
        bench = 'zz500_CLOSE'

    # read the stock data for later use
    f = open(stocklist_dir,'rb')
    stockLst = pickle.load(f)
    StockData = pd.DataFrame()
    f.close()

    with trange(len(stockLst)) as stock_bar:    
        for i in stock_bar:
            stockDF = stockLst[i]
            stock_id = stockDF['id'].iloc[0]
            stock_bar.set_description('Processing stock number %s'%(stock_id))
            # We only need the stock data within the backtest period
            stockDF = stockDF.loc[(stockDF['tdate'] >= start_date) & (stockDF['tdate'] <= end_date)].copy()
            # # if # of data is not enough for the requirement, ignoring it
            # if stockDF.shape[0] < len(timeLst):
            #     continue
            if stockPool != 'allAshare':
                # only choose stocks in a certain stock index
                if stockPool == 'hs300':
                    if stockDF['member'].tolist().count(1)<=benchPerc*stockDF['member'].shape[0]:
                        continue
                elif stockPool == 'zz500':
                    if stockDF['member'].tolist().count(2)<=benchPerc*stockDF['member'].shape[0]:
                        continue
                elif stockPool == 'zz1000':
                    if stockDF['member'].tolist().count(3)<=benchPerc*stockDF['member'].shape[0]:
                        continue
                elif stockPool == 'othershare':
                    if stockDF['member'].tolist().count(4)<=benchPerc*stockDF['member'].shape[0]:
                        continue
                elif stockPool == 'Top1800':
                    if stockDF['member'].tolist().count(1)<=benchPerc*stockDF['member'].shape[0] and \
                        stockDF['member'].tolist().count(2)<=benchPerc*stockDF['member'].shape[0] and \
                        stockDF['member'].tolist().count(3)<=benchPerc*stockDF['member'].shape[0]:
                        continue

            # if the stock satisfies all the requirements, we add it to the stock pool
            stockDF.rename(columns={'tdate': 'date'}, inplace=True)
            if not stockDF.empty:
                # calculate the abnormal return within the window
                stockDF['AR'] = stockDF['close'].shift(periods=-forward_window).fillna(method='bfill') / stockDF['close'].shift(periods=1).fillna(method='ffill')\
                    - stockDF[bench].shift(periods=-forward_window).fillna(method='bfill') / stockDF[bench].shift(periods=1).fillna(method='ffill')
                StockData = StockData.append(stockDF[['date', 'id', 'AR'] + features])

    StockData.reset_index(drop=True, inplace=True)

    # store processed stock data
    dir = r'%s\\Stock_Data\\Stock_Data_%s_%s_%s_%s.pkl' % (save_dir, start_date, end_date, back_window, forward_window)
    with open(dir, 'wb') as handle:
        pickle.dump(StockData, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.close()

    return None

# read notice data and find the corresponding report data
def read_notice_data(stockData: pd.DataFrame, notice_dir: str, report_dir: str, save_dir: str, timeLst: np.array, 
    report_columns: list, window: int=5):

    stockData.rename(columns={'tdate': 'date'}, inplace=True)
    stockData = stockData[['id', 'date', 'AR']]

    if not os.path.exists(r'%s\\Notice_Data' % save_dir):
        os.makedirs(r'%s\\Notice_Data' % save_dir)

    NoticeLst = os.listdir(notice_dir)
    # read the profit notice data by the order of dates (we only need the data of stock ID and date)
    with trange(len(NoticeLst)) as notice_bar:    
        for i in notice_bar:
            date_i = int(NoticeLst[i][-13:-5])
            notice_bar.set_description('Processing notice on date %s' % date_i)

            if date_i in timeLst:
                if os.path.exists(r'%s\\Notice_Data\\Notice_%s.csv' % (save_dir, date_i)):
                    continue

                # read notice data on date i
                dir = notice_dir + '/' + NoticeLst[i]
                notice_df = pd.read_excel(dir)

                if notice_df.empty:
                    continue

                notice_df.rename(columns={'S_INFO_WINDCODE': 'id', 'S_PROFITNOTICE_DATE': 'date'}, inplace=True)
                notice_df = notice_df[['id', 'date']].copy()
                notice_df['id'] = notice_df['id'].apply(lambda x: int(x[:6]))
                notice_df['date'] = notice_df['date'].astype(np.int32)

                # get the corresponding abnormal return data
                notice_df = pd.merge(notice_df, stockData, on=['id', 'date'], how='left')

                # match the report data in the following x dates
                temp_num = notice_df.shape[0]
                date_idx = np.where(date_i == timeLst)[0][0]
                try:
                    temp_time = timeLst[date_idx:date_idx+window]
                except:
                    temp_time = timeLst[date_idx:]

                temp_date = []
                for d in temp_time:
                    temp_date += [d] * temp_num

                notice_df = pd.concat([notice_df] * len(temp_time), ignore_index=True)
                notice_df['date'] = temp_date

                ReportDF = pd.DataFrame()
                for d in temp_time:
                    report_df = pd.read_excel(r'%s\\GOGOAL_CMB_REPORT_RESEARCH_%s.xlsx' % (report_dir, d))

                    if report_df.empty:
                        continue

                    report_df.rename(columns={'CODE': 'id', 'CREATE_DATE': 'date'}, inplace=True)
                    # report_df = report_df[report_df['ATTENTION_NAME'].apply(lambda x: '首' not in x)]
                    report_df = report_df[['id', 'date'] + report_columns].copy()
                    report_df['id'] = report_df['id'].astype(np.int32)
                    report_df['date'] = report_df['date'].astype(str).apply(lambda x: x[:4]+x[5:7]+x[8:10]).astype(np.int32)
                    ReportDF = ReportDF.append(report_df)

                if ReportDF.empty:
                    continue

                ReportDF.reset_index(drop=True, inplace=True)

                notice_df = pd.merge(notice_df, ReportDF, on=['id', 'date'], how='left')

                # drop if both title and content are nan
                notice_df.dropna(subset=['TITLE', 'CONTENT'], how='all', inplace=True)
                notice_df.to_csv(r'%s\\Notice_Data\\Notice_%s.csv' % (save_dir, date_i), index=False)

# read express data and find the corresponding report data
def read_express_data(stockData: pd.DataFrame, express_dir: str, report_dir: str, save_dir: str, timeLst: np.array, 
    report_columns: list, window: int=5):

    stockData.rename(columns={'tdate': 'date'}, inplace=True)
    stockData = stockData[['id', 'date', 'AR']]

    if not os.path.exists(r'%s\\Express_Data' % save_dir):
        os.makedirs(r'%s\\Express_Data' % save_dir)

    ExpressLst = os.listdir(express_dir)
    # read the profit express data by the order of dates (we only need the data of stock ID and date)
    with trange(len(ExpressLst)) as express_bar:    
        for i in express_bar:
            date_i = int(ExpressLst[i][-13:-5])
            express_bar.set_description('Processing express on date %s' % date_i)

            if date_i in timeLst:
                if os.path.exists(r'%s\\Express_Data\\Express_%s.csv' % (save_dir, date_i)):
                    continue

            if date_i in timeLst:
                dir = express_dir + '/' + ExpressLst[i]
                express_df = pd.read_excel(dir)

                if express_df.empty:
                    continue

                express_df.rename(columns={'S_INFO_WINDCODE': 'id', 'ANN_DT': 'date'}, inplace=True)
                express_df = express_df[['id', 'date']].copy()
                express_df['id'] = express_df['id'].apply(lambda x: int(x[:6]))
                express_df['date'] = express_df['date'].astype(np.int32)

                express_df = pd.merge(express_df, stockData, on=['id', 'date'], how='left')

                temp_num = express_df.shape[0]
                date_idx = np.where(date_i == timeLst)[0][0]
                try:
                    temp_time = timeLst[date_idx:date_idx+window]
                except:
                    temp_time = timeLst[date_idx:]

                temp_date = []
                for d in temp_time:
                    temp_date += [d] * temp_num

                express_df = pd.concat([express_df] * len(temp_time), ignore_index=True)
                express_df['date'] = temp_date

                ReportDF = pd.DataFrame()
                for d in temp_time:
                    report_df = pd.read_excel(r'%s\\GOGOAL_CMB_REPORT_RESEARCH_%s.xlsx' % (report_dir, d))
                    
                    if report_df.empty:
                        continue

                    report_df.rename(columns={'CODE': 'id', 'CREATE_DATE': 'date'}, inplace=True)
                    # report_df = report_df[report_df['ATTENTION_NAME'].apply(lambda x: '首' not in x)]
                    report_df = report_df[['id', 'date'] + report_columns].copy()
                    report_df['id'] = report_df['id'].astype(np.int32)
                    report_df['date'] = report_df['date'].astype(str).apply(lambda x: x[:4]+x[5:7]+x[8:10]).astype(np.int32)
                    ReportDF = ReportDF.append(report_df)

                if ReportDF.empty:
                    continue

                express_df = pd.merge(express_df, ReportDF, on=['id', 'date'], how='left')

                express_df.dropna(subset=['TITLE', 'CONTENT'], how='all', inplace=True)
                express_df.to_csv(r'%s\\Express_Data\\Express_%s.csv' % (save_dir, date_i), index=False)

# read statement data and find the corresponding report data
def read_statement_data(stockData: pd.DataFrame, state_dir: str, report_dir: str, save_dir: str, timeLst: np.array, 
    report_columns: list, window: int=5):

    if not os.path.exists(r'%s\\Statement_Data' % save_dir):
        os.makedirs(r'%s\\Statement_Data' % save_dir)

    stateData = pd.read_csv(state_dir)
    stateData.rename(columns={'Unnamed: 0': 'id'}, inplace=True)
    stateData.set_index(keys='id', inplace=True)
    stateData = stateData.unstack(level=0)
    stateData.dropna(inplace=True)
    stateData.name = 'date'
    idLst = [idx[1] for idx in stateData.index]
    state_df = pd.DataFrame({'id': idLst, 'date': stateData.values})

    # turn the id with type of object into nan and then drop them (such stocks should not be in the stock pool)
    def to_int(x):
        try:
            x = int(x[:6])
        except:
            return np.nan
        return x

    # read the financial statement data (we only need the data of stock ID and date)
    state_df['id'] = state_df['id'].apply(to_int)
    state_df.dropna(inplace=True)
    state_df['id'] = state_df['id'].astype(np.int32)
    state_df['date'] = state_df['date'].astype(str).apply(lambda x: x.split('/'))
    state_df['date'] = state_df['date'].apply(lambda x: [int(i) for i in x])
    state_df['date'] = state_df['date'].apply(lambda x: '%d%02d%02d' % (x[0], x[1], x[2]))
    state_df['date'] = state_df['date'].astype(np.int32)
    state_df = state_df[state_df['date'].isin(timeLst)].copy()

    dateLst = np.sort(pd.unique(state_df['date']))

    # read the profit express data by the order of dates (we only need the data of stock ID and date)
    with trange(len(dateLst)) as state_bar:    
        for i in state_bar:
            date_i = dateLst[i]
            state_bar.set_description('Processing statement on date %s' % date_i)

            if os.path.exists(r'%s\\Statement_Data\\Statement_%s.csv' % (save_dir, date_i)):
                continue

            state_i = state_df.loc[state_df['date'] == date_i]
            state_i = pd.merge(state_i, stockData, on=['id', 'date'], how='left')

            temp_num = state_i.shape[0]
            date_idx = np.where(date_i == timeLst)[0][0]
            try:
                temp_time = timeLst[date_idx:date_idx+window]
            except:
                temp_time = timeLst[date_idx:]

            temp_date = []
            for d in temp_time:
                temp_date += [d] * temp_num

            state_i = pd.concat([state_i] * len(temp_time), ignore_index=True)
            state_i['date'] = temp_date

            ReportDF = pd.DataFrame()
            for d in temp_time:
                report_df = pd.read_excel(r'%s\\GOGOAL_CMB_REPORT_RESEARCH_%s.xlsx' % (report_dir, d))
                
                if report_df.empty:
                    continue

                report_df.rename(columns={'CODE': 'id', 'CREATE_DATE': 'date'}, inplace=True)
                # report_df = report_df[report_df['ATTENTION_NAME'].apply(lambda x: '首' not in x)]
                report_df = report_df[['id', 'date'] + report_columns].copy()
                report_df['id'] = report_df['id'].astype(np.int32)
                report_df['date'] = report_df['date'].astype(str).apply(lambda x: x[:4]+x[5:7]+x[8:10]).astype(np.int32)
                ReportDF = ReportDF.append(report_df)

            if ReportDF.empty:
                continue

            state_i = pd.merge(state_i, ReportDF, on=['id', 'date'], how='left')

            state_i.dropna(subset=['TITLE', 'CONTENT'], how='all', inplace=True)
            state_i.dropna(subset=['AR'], inplace=True)
            state_i.to_csv(r'%s\\Statement_Data\\Statement_%s.csv' % (save_dir, date_i), index=False)


def tokenize_data(timeLst: np.array, save_dir: str, start_date: int, end_date: int, back_window: int=1,
    forward_window: int=1):

    if not os.path.exists(r'%s\\Token_Data\\back_%s_forward_%s' % (save_dir, back_window, forward_window)):
        os.makedirs(r'%s\\Token_Data\\back_%s_forward_%s' % (save_dir, back_window, forward_window))

    timeLst = timeLst[(timeLst >= start_date) & (timeLst <= end_date)]
    num_date = len(timeLst)

    with trange(num_date) as time_bar:    
        for i in time_bar:
            date_i = timeLst[i]

            if os.path.exists(r'%s\\Token_Data\\back_%s_forward_%s\\Token_Data_%s.csv' % (save_dir, back_window, forward_window, date_i)):
                continue

            if not os.path.exists(r'%s\\Notice_Data\\Notice_%s.csv' % (save_dir, date_i)) and \
                not os.path.exists(r'%s\\Express_Data\\Express_%s.csv' % (save_dir, date_i)):
                continue

            Data = pd.DataFrame()

            time_bar.set_description('Processing data on date %s' % (date_i))

            try:
                notice_i = pd.read_csv(r'%s\\Notice_Data\\Notice_%s.csv' % (save_dir, date_i))
                Data = Data.append(notice_i)
            except:
                pass

            try:
                express_i = pd.read_csv(r'%s\\Express_Data\\Express_%s.csv' % (save_dir, date_i))
                Data = Data.append(express_i)
            except:
                pass

            try:
                state_i = pd.read_csv(r'%s\\Statement_Data\\Statement_%s.csv' % (save_dir, date_i))
                Data = Data.append(state_i)
            except:
                pass

            if Data.empty:
                continue

            # if a stock has research data from both profit notice/express and financial statement, we drop the one from financial statement
            Data.reset_index(drop=True, inplace=True)
            Data['Year'] = Data['date'].apply(lambda x: int(str(x)[:4]))
            Data['Month'] = Data['date'].apply(lambda x: int(str(x)[4:6]))

            # tokenize the corpus by jieba
            Data['TITLE'] = Data['TITLE'].apply(tokenize)
            Data['CONTENT'] = Data['CONTENT'].apply(tokenize)
            Data.dropna(subset=['TITLE', 'CONTENT'], how='all', inplace=True)
            Data.drop_duplicates(subset=['id', 'date', 'TITLE', 'CONTENT'], inplace=True)
            Data.reset_index(drop=True, inplace=True)

            Data.to_csv(r'%s\\Token_Data\\back_%s_forward_%s\\Token_Data_%s.csv' % (save_dir, back_window, forward_window, date_i), index=False)

    return None

# tokenize the corpus data into words
def tokenize(corpus):
    if not pd.isnull(corpus):
        valid_words = ['n', 'nz', 'v', 'vd', 'vn', 'l', 'a', 'd']
        seg_list = jieba.posseg.lcut(corpus, use_paddle=True)
        content = [x for x, y in seg_list if y in valid_words]
    else:
        content = []
    new_corpus = ' '.join(content)
    return new_corpus

# get features of text data by counter
def get_count_feature(df, title_features=200, content_features=1000):

    title = df['TITLE'].values
    title_vectorizer = CountVectorizer(max_features=title_features)
    title = title_vectorizer.fit_transform(title).todense()
    title = np.log2(title + 1)

    content = df['CONTENT'].values
    content_vectorizer = CountVectorizer(max_features=content_features)
    content = content_vectorizer.fit_transform(content).todense()
    content = np.log2(content + 1)

    X = np.concatenate([title, content], axis=1)

    return X, title_vectorizer.get_feature_names(), content_vectorizer.get_feature_names()

# get SUE factor
def get_factor(model, df, title_vocab, content_vocab, title_features=100, content_features=500, delay_rate=0.95):
    title = df['TITLE'].values
    content = df['CONTENT'].values
    vectorizer_title = CountVectorizer(max_features=title_features, vocabulary=title_vocab)
    vectorizer_content = CountVectorizer(max_features=content_features, vocabulary=content_vocab)
    title = vectorizer_title.fit_transform(title).todense()
    content = vectorizer_content.fit_transform(content).todense()
    title = np.log2(title+1)
    content = np.log2(content+1)
    # concatenate the features from title and content
    X = np.concatenate([title, content], axis=1)
    Y = df.label.values
    # get the predicted probability
    prob = model.predict_proba(X)
    # get the accuracy
    pred_Y = model.predict(X)
    accuracy = accuracy_score(pred_Y, Y)
    # get factor SUE based on the probability
    prob_low = np.log2(prob[:, 0] / (1 - prob[:, 0])).reshape((-1, 1))
    prob_high = np.log2(prob[:, -1] / (1 - prob[:, -1])).reshape((-1, 1))
    SUE = prob_high - prob_low
    # decay the factor SUE
    delay = df['remain_days'].values
    trans_delay = lambda x: delay_rate ** x
    delay = np.vectorize(trans_delay)(delay).reshape((-1, 1))

    return np.multiply(SUE, delay), accuracy

# set label based on the rank of return
def get_label(df, thres: float=0.3):
    df['label'] = 1
    thres_low, thres_high = np.percentile(df['AR'], 100*thres), np.percentile(df['AR'], 100*(1-thres))
    df['label'].loc[df['AR'] <= thres_low] = 0
    df['label'].loc[df['AR'] >= thres_high] = 2
    return df

# train and test the machine learning model
def train_and_test(model, save_dir: str, params: dict, features: list, start_year: int, end_year: int,
    timeLst: list, back_window: int=1, forward_window: int=1, hyperopt: bool=False) -> pd.DataFrame:
    # set parameters for the model
    # model.set_params(num_class=3, n_jobs=8)
    model.set_params(**params)

    if not os.path.exists(r'%s\\Model\\back_%s_forward_%s' % (save_dir, back_window, forward_window)):
        os.makedirs(r'%s\\Model\\back_%s_forward_%s' % (save_dir, back_window, forward_window))

    if not os.path.exists(r'%s\\SUE\\%s' % (save_dir, model.name)):
        os.makedirs(r'%s\\SUE\\%s' % (save_dir, model.name))

    Factor = pd.DataFrame()
    accuracyLst = []
    uniform_dict = {}
    int_dict = {}
    choice_dict = {'learning_rate': [0.025, 0.05, 0.075], 'max_depth': [3, 5], 'sub_sample': [0.8, 0.85, 0.9, 0.95]}

    for y in range(start_year, end_year-1):
        print('Processing training data on %s - %s, test data on %s' % (y, y+1, y+2))
        # get the training and test data
        train_time = timeLst[(timeLst >= int('%d0101' % (y))) & (timeLst <= int('%d1231' % (y+1)))]
        test_time = timeLst[(timeLst >= int('%d0101' % (y+2))) & (timeLst <= int('%d1231' % (y+2)))]

        train_df, test_df = pd.DataFrame(), pd.DataFrame()

        with trange(len(train_time)) as train_bar:    
            for i in train_bar:
                date_i = train_time[i]
                train_bar.set_description('Loading training data on date %s' % (date_i))
                try:
                    temp_df = pd.read_csv(r'%s\\Token_Data\\back_%s_forward_%s\\Token_Data_%s.csv' % (save_dir, back_window, forward_window, date_i))
                except:
                    temp_df = pd.DataFrame()
                train_df = train_df.append(temp_df)

        train_df.reset_index(drop=True, inplace=True)

        with trange(len(test_time)) as test_bar:    
            for i in test_bar:
                date_i = test_time[i]
                test_bar.set_description('Loading testing data on date %s' % (date_i))
                try:
                    temp_df = pd.read_csv(r'%s\\Token_Data\\back_%s_forward_%s\\Token_Data_%s.csv' % (save_dir, back_window, forward_window, date_i))
                except:
                    temp_df = pd.DataFrame()
                test_df = test_df.append(temp_df)

        test_df.reset_index(drop=True, inplace=True)

        # get the label of training data
        train_df.dropna(subset=['AR'], inplace=True)
        train_df.reset_index(drop=True, inplace=True)
        train_df = get_label(train_df)
        train_df['TITLE'] = train_df['TITLE'].fillna('')
        train_df['CONTENT'] = train_df['CONTENT'].fillna('')

        # get the features and train the model
        train_X, title_vocab, content_vocab = get_count_feature(train_df)
        train_Y = train_df.label.values

        if os.path.exists(r'%s\\Model\\back_%s_forward_%s\\%s_%s_%s.m' % (save_dir, back_window, \
            forward_window, model.name, y, y+1)):
            modelFitted = joblib.load(r'%s\\Model\\back_%s_forward_%s\\%s_%s_%s.m' % (save_dir, back_window, \
                forward_window, model.name, y, y+1))

        else:
            if not hyperopt:
                modelFitted = model.fit(train_X, train_Y)
            else:
                optparams = model.hyperopt(train_X, train_Y, uniform_dict, int_dict, choice_dict)
                modelFitted = model.set_params(**optparams)
                modelFitted = modelFitted.fit(train_X, train_Y)

            joblib.dump(modelFitted, r'%s\\Model\\back_%s_forward_%s\\%s_%s_%s.m' % (save_dir, back_window, \
                forward_window, model.name, y, y+1))
            
        # get the label of test data
        test_df.dropna(subset=['AR'], inplace=True)
        test_df.reset_index(drop=True, inplace=True)

        # record the time interval between current date and the end of the month
        dayLst = []

        for j in range(test_df.shape[0]):
            dateStr = str(test_df['date'].iloc[j])
            days = month_to_days[int(dateStr[4:6]) % 12]
            dayLst.append(days - int(dateStr[-2:]))

        test_df['remain_days'] = dayLst

        test_df = get_label(test_df)

        test_df['TITLE'] = test_df['TITLE'].fillna('')
        test_df['CONTENT'] = test_df['CONTENT'].fillna('')

        # get factor SUE
        SUE, accuracy = get_factor(modelFitted, test_df, title_vocab, content_vocab)
        accuracyLst.append(accuracy)
        test_df['SUE_%s' % model.name] = SUE
        test_df['Year'] = test_df['Year'].astype(str)
        test_df['Month'] = test_df['Month'].apply(lambda x: '%02d' % x)
        test_df['date'] = test_df['Year'] + test_df['Month']
        Factor = Factor.append(test_df[['id', 'SUE_%s' % model.name, 'date']+features])

    Factor = Factor[['id', 'date', 'SUE_%s' % model.name]+features].copy()
    Factor = Factor.groupby(by=['id', 'date']).mean()
    Factor.reset_index(drop=False, inplace=True)
    Factor.date = Factor.date.apply(lambda x: int(str(x)[:6]))
    Factor.to_csv(r'%s\\SUE\\%s\\SUE_%s_raw.csv' % (save_dir, model.name, model.name), index=False)


def clear_factor(factor_dir: str, save_dir: str, model, start_year: int, end_year: int, train_window: int=3):
    Factor = pd.read_csv(factor_dir)
    Result = pd.DataFrame()
    Last_stock_pool = pd.DataFrame()

    MonthLst = np.array(['%d%02d' % (y, m) for y in range(start_year, end_year+1) for m in range(1, 13)])
    temp = [i for i in range(1,11)]

    for y in range(start_year, end_year-1):
        for m in range(1, 13):

            time_i = '%d%02d' % (y+2, m)
            idx = np.where(MonthLst == time_i)[0][0]

            if y == start_year and m == 1:
                Last_stock_pool = Factor.loc[Factor.date == int(time_i)]
                Result = Result.append(Last_stock_pool)
                continue

            if m == 4:
                temp_time = MonthLst[idx-2:idx+1]
                Last_stock_pool = pd.DataFrame()
                copy_last = False
            elif m == 8:
                temp_time = MonthLst[idx-1:idx+1]
                Last_stock_pool = pd.DataFrame()
                copy_last = False
            elif m == 10:
                temp_time = [MonthLst[idx]]
                Last_stock_pool = pd.DataFrame()
                copy_last = False
            else:
                temp_time = MonthLst[idx-train_window+1:idx+1]
                copy_last = True

            # stock data in this month
            df_i = Factor.loc[Factor['date'] == int(time_i)].copy()

            if not Last_stock_pool.empty:
                temp_idx = Last_stock_pool.id.tolist()
                # the stocks which have factor value in both last month and this month
                df_i_1 = df_i[df_i.id.isin(temp_idx)].copy()
                # the stocks which doesn't have factor value in last month but have in this month
                df_i_2 = df_i[~df_i.id.isin(temp_idx)].copy()
                # the stocks which has factor value in last month but doesn't have in this month
                temp_hold = Last_stock_pool[~Last_stock_pool.id.isin(df_i_1.id.tolist())].copy()
                temp_hold['date'] = int(time_i)

                if copy_last:
                    Result = Result.append(temp_hold)

            temp_time = [int(t) for t in temp_time]
            temp = Factor[Factor['date'].isin(temp_time)].groupby('id').agg({'SUE_%s' % model.name: np.mean})
            temp.reset_index(drop=False, inplace=True)
            temp['date'] = int(time_i)
            Result = Result.append(temp)
            Last_stock_pool = temp

    Result.drop_duplicates(subset=['id', 'date'], inplace=True)
    Result.sort_values(by=['id', 'date'], inplace=True)
    Result.reset_index(drop=True, inplace=True)
    Result.to_csv(r'%s\\SUE\\%s\\SUE_%s.csv' % (save_dir, model.name, model.name), index=False)

# fill the NA value by rolling average
def fill_na(df, window=5):
    for col in df.columns:
        df[col]= df[col].fillna(df[col].rolling(window+1, min_periods=1).mean())
    return df


# clip the extreme values within 3 standard deviation 
def clip_extreme(df):
    for col in df.columns:
        avg = df[col].mean()
        standard = df[col].std()
        df[col].clip(avg - 3*standard, avg + 3*standard, inplace=True)
    return df


# normalize all the features
def normalize(df):
    for col in df.columns:
        avg = df[col].mean()
        standard = df[col].std()
        df[col] = (df[col] - avg) / standard
    return df


"""more utils"""
# update the columns of dates to fulfill the requirement of backtest system
def update_time(df: pd.DataFrame, time_dir: str, start_date: int, end_date: int):
    TradeLst = read_tradeDate_data(time_dir, start_date, end_date)
    TradeLst = np.array(TradeLst)
    df['this_month'] = 0
    df['next_month'] = 0
    MonthLst = np.sort(pd.unique(df['date']))
    for t in MonthLst:
        time_i = int(str(t)+'99')
        this_month = TradeLst[TradeLst <= time_i][-1]
        next_month = TradeLst[TradeLst >= time_i][0]
        df['this_month'].loc[df.date == t] = this_month
        df['next_month'].loc[df.date == t] = next_month
    return df

# update the factor
# logic of updating: For April, getting factor based on data from February, Marth, and April;
# For August, getting factor based on data from July and August
# For October, getting factor based on data from October
# For other months, first copy the factor from its last month, then getting factor based on data from the latest three months
def update_factor(df: pd.DataFrame):
    MonthLst = np.sort(pd.unique(df['date']))
    Factor = pd.DataFrame()
    Last_stock_pool = pd.DataFrame()
    for i in range(1, len(MonthLst)):
        time_i = MonthLst[i]
        
        if str(time_i)[-2:] == '04' or str(time_i)[-2:] == '08' or str(time_i)[-2:] == '10':
            Last_stock_pool = pd.DataFrame()
            continue

        if Last_stock_pool.empty:
            Last_stock_pool = df.loc[df.date == MonthLst[i-1]].copy()
            Factor = Factor.append(Last_stock_pool)

        df_i = df.loc[df.date == time_i].copy()
        temp_idx = Last_stock_pool.id.tolist()
        df_i_1 = df_i[df_i.id.isin(temp_idx)].copy()
        df_i_2 = df_i[~df_i.id.isin(temp_idx)].copy()
        temp_hold = Last_stock_pool[~Last_stock_pool.id.isin(df_i_1.id.tolist())].copy()
        temp_hold['date'] = time_i
        df_i_1 = df_i_1.append(temp_hold).append(df_i_2)
        Last_stock_pool = df_i_1
        Factor = Factor.append(df_i_1)
    Factor.sort_values(by=['id', 'date'], inplace=True)
    return Factor