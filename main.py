from utils import *


if __name__ == "__main__":

    # directory
    save_dir = 'E:/实习/南方202209/深度学习多头组目标函数'
    stocklist_dir = 'F:/南方基金实习/数据/stockList_all_5019_20050104_20221107.pkl'
    intraDay_dir = 'F:/南方基金实习/数据/IntraDayFeatureWideMat'
    factor20_dir = 'F:/南方基金实习/数据/Factor20'
    member_dir = 'F:/南方基金实习/数据/BenchPool/000905.SHMember_LongMat_1599_20070115_20230131.pkl'
    feature_dir = 'F:/南方基金实习/模型/树模型应用/Data/Factor20'
    # feature_dir = 'F:/南方基金实习/模型/树模型应用/Data/IntraDay'
    trDays_dir = 'F:/南方基金实习/数据/TrDate/TrDateAll_8803_19901219_20261231.pkl'
    bench_dir = 'E:/实习/南方202209/数据'

    # string parameters
    benchmark = 'zz500'
    # loss = 'Closs'
    # loss = 'Closs_sigmoid'
    # loss = 'Closs_explained'
    # loss = 'Lloss'
    # loss = 'ListMLE'
    # loss = 'ListNet'
    # loss = 'RankNet'
    # loss = 'LambdaLoss'
    # loss = 'ApproxNDCG'
    # loss = 'BinaryListNet'
    loss = 'TRR'
    # loss = 'MSE'
    short = 'none'
    # name = 'IntraDay'
    name = 'Factor20'

    # int parameters
    start_date = 20100101
    end_date = 20211231
    backtest_start_date = 20170103
    backtest_end_date = 20211229
    T = 5
    start_year = 2012
    end_year = 2021
    train_window = 5
    test_window = 1
    N = 1
    rank_to_use = 50

    # others
    return_dir = f'F:/南方基金实习/模型/树模型应用/Data/Return/allAshare/T+1_T+6'
    modelLst = ['LSTM']
    model_name = 'DNN'

    # # Step 1: save factor data for further use
    # read_factor_data_pkl(save_dir, intraDay_dir, member_dir, start_date, end_date, name, benchmark)
    # read_factor_data_csv(save_dir, factor20_dir, member_dir, start_date, end_date, name, benchmark)

    # # Step 2: save the return data for further use
    # read_return_data(stocklist_dir, save_dir, start_date, end_date, T)
    # read_price_data(stocklist_dir, save_dir, start_date, end_date)

    # Step 3: save the whole feature data
    get_rolling_feature_data(save_dir, feature_dir, trDays_dir, return_dir, member_dir, modelLst, loss, start_year, end_year, 
                             train_window, test_window, benchmark, name, N_STOCKS)
    
    # # Step 4: do the prediction and backtesting
    # get_predict(save_dir, trDays_dir, modelLst, loss, start_date, end_date, short, N, rank_to_use, T, start_year, end_year, 
    #             train_window, test_window, benchmark, name)

    # # Step 5: do backtesting based on quantstats
    # qs_backtest_bench(save_dir, model_name, loss, short, rank_to_use, 
    #     backtest_start_date, backtest_end_date, benchmark, name)