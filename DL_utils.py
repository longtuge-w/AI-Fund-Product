import os
import sys
import pickle
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
from sklearn.model_selection import train_test_split
import torch
from torch.nn.utils.rnn import pad_sequence
from torch import optim, nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import gensim
from torchtext.vocab import build_vocab_from_iterator


# used to ignore the warnings
warnings.filterwarnings("ignore")

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Variables
EMBEDDING_DIM = 300
MAX_SEQUENCE_LEN = 200
TRAIN_VALID_RATE = 0.2
BATCH_SIZE = 1
LEARNING_RATE = 1e-4
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-5
MAX_EPOCH = 1
NUM_CLASSES = 3
RANDOM_STATE = 1

# Variables for RNN only
N_LAYER = 5
HIDDEN_SIZE = 100
DROPOUT = 0.5

class Data(Dataset):
    """
    The simple Dataset object from torch that can produce reshuffled batchs of data
    """
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


class TestData(Dataset):
    """
    The simple Dataset object from torch that can produce reshuffled batchs of data
    """
    def __init__(self, X, y, r):
        self.X = X
        self.y = y
        self.r = r

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.X[index], self.y[index], self.r[index]


class EmbedLayer(nn.Module):
    """
    Embedding layer, used to transform sequence train data into the desired matrix. We need to define the 
    embedding matrix by some word embedding method like Glove

    Args:
        num_embedding (int): the total number of words we have, or to say the length of the vocabulary
        embedding_dim (int): The dimension of the embedded word
        embed_mat (tensor): The embedding matrix we want to serve as the weight of this layer
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, embed_mat: np.ndarray):
        super(EmbedLayer, self).__init__()
        # Define the embedding layer
        self.embed = nn.Embedding(num_embeddings, embedding_dim)
        self.embed.weight = nn.Parameter(torch.from_numpy(embed_mat).float(), requires_grad=False)

    def forward(self, x):
        x = self.embed(x)
        return x

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


def read_date_data(text_dir: str, start_date: int, end_date: int):
    # read the data for all dates
    timeLst = os.listdir(text_dir)
    timeLst = np.array(sorted([int(i[-13:-5]) for i in timeLst]))
    timeLst = timeLst[(timeLst >= start_date) & (timeLst <= end_date)]
    return timeLst


def read_stock_data(stocklist_dir: str, start_date: int, end_date: int, features: list, save_dir: str, stockPool: str='zz500',
    benchmark: str='zz500', benchPerc: float=0.5, back_window: int=1, forward_window: int=1, ):

    if not os.path.exists(r'%s\\Stock_Data' % save_dir):
        os.makedirs(r'%s\\Stock_Data' % save_dir)
    
    if os.path.exists(r'%s\\Stock_Data\\Stock_Data_%s_%s_%s_%s.pkl' % (save_dir, start_date, end_date, back_window, forward_window)):
        print('Required stock data already exists')
        return None

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
                stockDF['AR'] = stockDF['close'].shift(periods=-forward_window) / stockDF['close'].shift(periods=1)\
                    - stockDF[bench].shift(periods=-forward_window) / stockDF[bench].shift(periods=1)
                StockData = StockData.append(stockDF[['date', 'id', 'AR'] + features])

    StockData.reset_index(drop=True, inplace=True)

    dir = r'%s\\Stock_Data\\Stock_Data_%s_%s_%s_%s.pkl' % (save_dir, start_date, end_date, back_window, forward_window)
    with open(dir, 'wb') as handle:
        pickle.dump(StockData, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.close()

    return None


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
            notice_bar.set_description('Processing notice number %s'%(i))
            date_i = int(NoticeLst[i][-13:-5])

            if date_i in timeLst:
                if os.path.exists(r'%s\\Notice_Data\\Notice_%s.csv' % (save_dir, date_i)):
                    continue

                dir = notice_dir + '/' + NoticeLst[i]
                notice_df = pd.read_excel(dir)
                report_df = pd.read_excel(r'%s\\GOGOAL_CMB_REPORT_RESEARCH_%s.xlsx' % (report_dir, date_i))

                if notice_df.empty or report_df.empty:
                    continue

                notice_df.rename(columns={'S_INFO_WINDCODE': 'id', 'S_PROFITNOTICE_DATE': 'date'}, inplace=True)
                notice_df = notice_df[['id', 'date']].copy()
                notice_df['id'] = notice_df['id'].apply(lambda x: int(x[:6]))
                notice_df['date'] = notice_df['date'].astype(np.int32)

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

                report_df.rename(columns={'CODE': 'id', 'CREATE_DATE': 'date'}, inplace=True)
                report_df = report_df[report_df['ATTENTION_NAME'].apply(lambda x: '首' not in x)]
                report_df = report_df[['id', 'date'] + report_columns].copy()
                report_df['id'] = report_df['id'].astype(np.int32)
                report_df['date'] = report_df['date'].astype(str).apply(lambda x: x[:4]+x[5:7]+x[8:10]).astype(np.int32)

                notice_df = pd.merge(notice_df, report_df, on=['id', 'date'], how='left')
                notice_df = pd.merge(notice_df, stockData, on=['id', 'date'], how='left')

                notice_df.to_csv(r'%s\\Notice_Data\\Notice_%s.csv' % (save_dir, date_i), index=False)


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
            express_bar.set_description('Processing adjust score report number %s'%(i))
            date_i = int(ExpressLst[i][-13:-5])

            if date_i in timeLst:
                if os.path.exists(r'%s\\Express_Data\\Express_%s.csv' % (save_dir, date_i)):
                    continue

            if date_i in timeLst:
                dir = express_dir + '/' + ExpressLst[i]
                express_df = pd.read_excel(dir)
                report_df = pd.read_excel(r'%s\\GOGOAL_CMB_REPORT_RESEARCH_%s.xlsx' % (report_dir, date_i))

                if express_df.empty or report_df.empty:
                    continue

                express_df.rename(columns={'S_INFO_WINDCODE': 'id', 'ANN_DT': 'date'}, inplace=True)
                express_df = express_df[['id', 'date']].copy()
                express_df['id'] = express_df['id'].apply(lambda x: int(x[:6]))
                express_df['date'] = express_df['date'].astype(np.int32)

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

                report_df.rename(columns={'CODE': 'id', 'CREATE_DATE': 'date'}, inplace=True)
                report_df = report_df[report_df['ATTENTION_NAME'].apply(lambda x: '首' not in x)]
                report_df = report_df[['id', 'date'] + report_columns].copy()
                report_df['id'] = report_df['id'].astype(np.int32)
                report_df['date'] = report_df['date'].astype(str).apply(lambda x: x[:4]+x[5:7]+x[8:10]).astype(np.int32)

                express_df = pd.merge(express_df, report_df, on=['id', 'date'], how='left')
                express_df = pd.merge(express_df, stockData, on=['id', 'date'], how='left')

                express_df.to_csv(r'%s\\Express_Data\\Express_%s.csv' % (save_dir, date_i), index=False)


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
            notice_i = pd.read_csv(r'%s\\Notice_Data\\Notice_%s.csv' % (save_dir, date_i))
            express_i = pd.read_csv(r'%s\\Express_Data\\Express_%s.csv' % (save_dir, date_i))
            Data = Data.append(notice_i).append(express_i)

            # if a stock has research data from both profit notice/express and financial statement, we drop the one from financial statement
            Data.reset_index(drop=True, inplace=True)
            Data['Year'] = Data['date'].apply(lambda x: int(str(x)[:4]))
            Data['Month'] = Data['date'].apply(lambda x: int(str(x)[4:6]))
            # tokenize the corpus by jieba
            Data['TITLE'] = Data['TITLE'].apply(tokenize)
            Data['CONTENT'] = Data['CONTENT'].apply(tokenize)
            Data = Data.dropna(subset=['CONTENT'])
            Data.drop_duplicates(subset=['TITLE', 'CONTENT'], inplace=True)
            Data.reset_index(drop=True, inplace=True)

            Data.to_csv(r'%s\\Token_Data\\back_%s_forward_%s\\Token_Data_%s.csv' % (save_dir, back_window, forward_window, date_i), index=False)

    return None

# tokenize the corpus data into words
def tokenize(corpus):
    new_corpus = ''
    valid_words = ['n', 'nz', 'v', 'vd', 'vn', 'l', 'a', 'd']
    seg_list = jieba.posseg.lcut(corpus, use_paddle=True)
    content = [x for x, y in seg_list if y in valid_words]
    new_corpus = ' '.join(content)
    return new_corpus

# get feature by using counter
def get_feature(df, vocab):

    df['CORPUS'] = df['TITLE'] + df['CONTENT']
    df['CORPUS'] = df['CORPUS'].apply(lambda x: x.split())
    df['CORPUS'] = df['CORPUS'].apply(lambda x: torch.LongTensor(vocab(x)))

    return df['CORPUS'].tolist()

# get SUE factor
def get_factor(model, dataloader, delay_rate=0.95, eps=1e-5):

    # enumerate mini batches
    print('Evaluating ...')
    test_data_size = len(dataloader)
    test_dataiter = iter(dataloader)

    SUE = []
    Accuracy = 0
    # set the bar to check the progress
    with trange(test_data_size) as test_bar:
        for i in test_bar:
            test_bar.set_description('Evaluating batch %s'%(i+1))
            x_test, y_test, r_test = next(test_dataiter)
            x_test, y_test, r_test = x_test.to(device), y_test.to(device), r_test.to(device)

            # compute the model output
            y_pred = model(x_test)

            # calculate loss
            _, preds = torch.max(y_pred, 1)
            Accuracy += torch.sum(preds == y_test.data).item()

            # set information for the bar
            test_bar.set_postfix(accuracy=Accuracy/(BATCH_SIZE*(i+1)))

            y_pred = F.softmax(y_pred, dim=1)
            r_test = r_test.to(dtype=torch.float32)
            y_pred = y_pred.detach().cpu().apply_(lambda x: np.log((x + eps) / (1 - x + eps)))
            r_test = r_test.detach().cpu().apply_(lambda x: delay_rate ** x)
            temp = y_pred[:,-1] - y_pred[:,0]
            temp = torch.mul(temp, r_test).detach().cpu().tolist()
            SUE.extend(temp)

            del x_test, y_test, r_test
            torch.cuda.empty_cache()

    return SUE

# set label based on the rank of return
def get_label(df, thres: float=0.3):
    df['label'] = 1
    thres_low, thres_high = np.percentile(df['return'], 100*thres), np.percentile(df['return'], 100*(1-thres))
    df['label'].loc[df['return'] <= thres_low] = 0
    df['label'].loc[df['return'] >= thres_high] = 2
    return df


def evaluate(model, dataloader):

    # enumerate mini batches
    print('Evaluating ...')
    test_data_size = len(dataloader)
    test_dataiter = iter(dataloader)
    model.eval()

    Accuracy = 0

    # set the bar to check the progress
    with trange(test_data_size) as test_bar:
        for i in test_bar:
            test_bar.set_description('Evaluating batch %s'%(i+1))
            x_test, y_test = next(test_dataiter)
            x_test, y_test = x_test.to(device), y_test.to(device)

            # compute the model output
            y_pred = model(x_test)

            # calculate loss
            _, preds = torch.max(y_pred, 1)
            Accuracy += torch.sum(preds == y_test.data).item()

            # set information for the bar
            test_bar.set_postfix(accuracy=Accuracy/(BATCH_SIZE*(i+1)))

            del x_test, y_test
            torch.cuda.empty_cache()

        return Accuracy/(BATCH_SIZE*(i+1))
    

def train(model, train_dataloader, valid_dataloader):
    # set the criterion
    criterion = nn.CrossEntropyLoss()

    best_model = None
    highest_acc = 0

    # set the optimizer
    # optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print('{} epochs to train: '.format(MAX_EPOCH))

    for epoch in range(1, MAX_EPOCH+1):

        # enumerate mini batches
        print('epoch {}/{}:'.format(epoch, MAX_EPOCH))
        train_data_size = len(train_dataloader)
        # train_data_size = 10
        train_dataiter = iter(train_dataloader)
        model.train()

        Accuracy = 0
        Total_loss = 0

        # set the bar to check the progress
        with trange(train_data_size) as train_bar:
            for i in train_bar:
                train_bar.set_description('Training batch %s'%(i+1))
                x_train, y_train = next(train_dataiter)
                x_train, y_train = x_train.to(device), y_train.to(device)

                # clear the gradients
                optimizer.zero_grad()

                # compute the model output
                y_pred = model(x_train)

                # calculate loss
                train_loss = criterion(y_pred, y_train)
                _, preds = torch.max(y_pred, 1)
                Accuracy += torch.sum(preds == y_train.data).item()
                Total_loss += train_loss.item()

                # credit assignment
                train_loss.backward(retain_graph=True)

                # update model weights
                optimizer.step()

                # set information for the bar
                train_bar.set_postfix(train_loss=Total_loss/(BATCH_SIZE*(i+1)), 
                    accuracy=Accuracy/(BATCH_SIZE*(i+1)))

                del x_train, y_train
                torch.cuda.empty_cache()

        model.eval()
        accuracy = evaluate(model, valid_dataloader)
        # choose the model with highest accuracy on the validation data
        if accuracy > highest_acc:
            best_model = model
        model.train()

    return best_model


def get_embed_mat(word2vec, vocab):
    # determine the value of embedding matrix by Glove
    embedding_matrix = []
    for i in range(len(vocab)):
        token = vocab.lookup_token(i)
        # get the corresponding vector of the token. If there is no such vector, we just assign a random value
        # with the same dimension
        try:
            embedding_vector = word2vec.get_vector(token, norm=True)
        except:
            embedding_vector = np.random.randn(EMBEDDING_DIM)
        
        embedding_matrix.append(embedding_vector)

    embedding_matrix = np.vstack(embedding_matrix)

    return embedding_matrix

# train and test the machine learning model
def train_and_test(df, model, save_dir: str, word2vec_dir: str, params: dict, features: list, start_year: int, end_year: int,
    timeLst: list, back_window: int=1, forward_window: int=1, train_window: int=6):

    if not os.path.exists(r'%s\\Model\\back_%s_forward_%s' % (save_dir, back_window, forward_window)):
        os.makedirs(r'%s\\Model\\back_%s_forward_%s' % (save_dir, back_window, forward_window))

    word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vec_dir, binary=True, unicode_errors='ignore')
    MonthLst = np.array(['%d%02d' % (y, m) for y in range(start_year, end_year+1) for m in range(1, 13)])
    temp = [i for i in range(1,11)]
    Factor = pd.DataFrame()

    # build vocabulary for the text
    print('Loading the vocabulary ...')
    vocabLst = [text.split() for text in df.CONTENT]
    vocab = build_vocab_from_iterator(vocabLst, specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])

    print('Loading the embedding matrix ...')
    embedding_matrix = get_embed_mat(word2vec, vocab)

    for y in range(start_year, end_year-1):
        # get the training and test data
        train_time = timeLst[(timeLst >= int('%d0101' % (y))) & (timeLst <= int('%d1231' % (y+1)))]
        test_time = timeLst[(timeLst >= int('%d0101' % (y+2))) & (timeLst <= int('%d1231' % (y+2)))]

        train_df, test_df = pd.DataFrame(), pd.DataFrame()

        with trange(len(train_time)) as train_bar:    
            for i in train_bar:
                date_i = train_time[i]
                train_bar.set_description('Loading training data on date %s'%(i))
                temp_df = pd.read_csv(r'%s\\Token_Data\\back_%s_forward_%s\\Token_Data_%s.csv' % (save_dir, back_window, forward_window, date_i))
                train_df.append(temp_df)

        train_df.reset_index(drop=True, inplace=True)

        with trange(len(test_time)) as test_bar:    
            for i in test_bar:
                date_i = test_time[i]
                test_bar.set_description('Loading testing data on date %s'%(i))
                temp_df = pd.read_csv(r'%s\\Token_Data\\back_%s_forward_%s\\Token_Data_%s.csv' % (save_dir, back_window, forward_window, date_i))
                train_df.append(temp_df)

        test_df.reset_index(drop=True, inplace=True)

        train_df.dropna(subset=['return'], inplace=True)
        train_df = get_label(train_df)

        train_df, valid_df = train_test_split(train_df, test_size=TRAIN_VALID_RATE, random_state=RANDOM_STATE)

        train_X = get_feature(train_df, vocab)
        valid_X = get_feature(valid_df, vocab)
        train_Y = train_df.label.values
        valid_Y = valid_df.label.values

        # pad first seq to desired length
        train_X[0] = torch.nn.ConstantPad1d((0, MAX_SEQUENCE_LEN - train_X[0].shape[0]), 0)(train_X[0])
        valid_X[0] = torch.nn.ConstantPad1d((0, MAX_SEQUENCE_LEN - valid_X[0].shape[0]), 0)(valid_X[0])

        # pad all seqs to desired length
        train_X = torch.t(pad_sequence(train_X))
        valid_X = torch.t(pad_sequence(valid_X))

        train_Y = torch.LongTensor(train_Y)
        valid_Y = torch.LongTensor(valid_Y)

        train_X = train_X[:,:MAX_SEQUENCE_LEN]
        valid_X = valid_X[:,:MAX_SEQUENCE_LEN]

        # embedding the sequence data
        Embed = EmbedLayer(len(vocab), EMBEDDING_DIM, embedding_matrix)

        train_X, valid_X = Embed(train_X), Embed(valid_X)

        # define data set and data loader
        train_Dataset = Data(train_X, train_Y)
        train_Dataloader = DataLoader(train_Dataset, batch_size=BATCH_SIZE, shuffle=False)
        valid_Dataset = Data(valid_X, valid_Y)
        valid_Dataloader = DataLoader(valid_Dataset, batch_size=BATCH_SIZE, shuffle=False)

        if os.path.exists(r'%s\\Model\\back_%s_forward_%s\\%s_%s_%s.m' % (save_dir, back_window, \
            forward_window, model.name, y, y+1)):
            modelFitted = joblib.load(r'%s\\Model\\back_%s_forward_%s\\%s_%s_%s.m' % (save_dir, back_window, \
                forward_window, model.name, y, y+1))
            modelFitted.to(device=device)
            modelFitted.eval()

        else:
            model.to(device=device)
            model.train()
            modelFitted = train(model, train_Dataloader, valid_Dataloader)
            modelFitted.eval()
            joblib.dump(modelFitted, r'%s\\Model\\back_%s_forward_%s\\%s_%s_%s.m' % (save_dir, back_window, \
                forward_window, model.name, y, y+1))

        for m in range(1, 13):
            print('Processing data in Year %s Month %s' % (y, m))
            idx = np.where(MonthLst == str(y+2)+str(m))[0][0]
            YearMonthLst = MonthLst[idx-2:idx+1]
            Test = pd.DataFrame()
            for i in range(len(YearMonthLst)):
                date = YearMonthLst[i]
                temp = test_df.loc[(test_df.Year == int(date[:4])) & (test_df.Month == int(date[4:]))].copy()

                # record the time interval between current date and the end of the month
                dayLst = []
                for j in range(temp.shape[0]):
                    dateStr = str(temp['date'].iloc[j])
                    if i == 0:
                        days = month_to_days[int(str(int(dateStr) + 2)[4:6]) % 12]
                        other_days = month_to_days[int(dateStr[4:6]) % 12] + month_to_days[int(str(int(dateStr) + 1)[4:6]) % 12]
                    elif i == 1:
                        days = month_to_days[int(str(int(dateStr) + 1)[4:6]) % 12]
                        other_days = month_to_days[int(dateStr[4:6]) % 12]
                    elif i == 2:
                        days = month_to_days[int(dateStr[4:6]) % 12]
                        other_days = 0
                    dayLst.append(days + other_days - int(dateStr[-2:]))

                temp['remain_days'] = dayLst

                Test = Test.append(temp)

            if Test.empty:
                continue

            Test.dropna(subset=['return'], inplace=True)
            Test = get_label(Test)
            test_X = get_feature(Test, vocab)
            test_r = Test.remain_days.values

            test_X[0] = torch.nn.ConstantPad1d((0, MAX_SEQUENCE_LEN - test_X[0].shape[0]), 0)(test_X[0])
            test_X = torch.t(pad_sequence(test_X))
            test_X = test_X[:,:MAX_SEQUENCE_LEN]
            test_X = Embed(test_X)

            test_Y = Test.label.values
            test_Y = torch.LongTensor(test_Y)

            test_r = torch.from_numpy(test_r)

            test_Dataset = TestData(test_X, test_Y, test_r)
            test_Dataloader = DataLoader(test_Dataset, batch_size=BATCH_SIZE, shuffle=False)

            SUE = get_factor(modelFitted, test_Dataloader)
            Test['SUE'] = SUE
            Test['date'] = '%d%02d' % (y+2, m)
            Factor = Factor.append(Test)

        del modelFitted
        torch.cuda.empty_cache()

    Factor = Factor[['id', 'date', 'SUE']].copy()
    Factor = Factor.groupby(by=['id', 'date']).mean()

    Factor.reset_index(drop=False, inplace=True)

    Factor.date = Factor.date.apply(lambda x: int(str(x)[:6]))
    Factor = Factor.groupby(by=['id', 'date']).mean()

    Factor.reset_index(drop=False, inplace=True)

    return Factor


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