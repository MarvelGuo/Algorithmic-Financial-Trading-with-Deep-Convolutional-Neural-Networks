"""
Created on 2020/3/28 9:48
Author: Xinyu Guo
Email: xyguo@bu.edu
IDE: PyCharm
"""

from sklearn.utils import compute_class_weight
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import os
import yfinance as yf
import numpy as np
import pandas as pd
import tqdm
from Tech_Indicator import *
from warnings import simplefilter
simplefilter(action='ignore', category=Warning)


class Data:
    def __init__(self, ticker, group_train_len=5, group_test_len=1):
        self.ticker = ticker
        self.group_train_len = group_train_len
        self.group_test_len = group_test_len
        self.price = None
        self.df_Tech = None
        self.ticker_path = self.check_file_path()

    def check_file_path(self):
        ticker_path = './Data/' + self.ticker
        if not os.path.exists('./Data'):
            os.mkdir('Data')
        if not os.path.exists(ticker_path):
            os.makedirs(ticker_path)
        return ticker_path

    def Load_Data(self, start_date, end_date, ticker=None):
        '''
        Used to read/download the data for a ticker in whole period
        If data is not in file, use yfinance to get data and save
        '''
        if ticker is None:
            ticker = self.ticker
        file_name = ticker + '_' + \
            start_date.replace('-', '') + '_' + end_date.replace('-', '')
        file_path = self.ticker_path + os.sep + file_name + '.csv'
        if os.path.exists(file_path):
            price_df = pd.read_csv(file_path, index_col=0, parse_dates=[0])
        else:
            price_df = yf.Ticker(ticker).history(start=start_date, end=end_date)[
                ['Open', 'High', 'Low', 'Close', 'Volume']]
            price_df.columns = [c.lower() for c in price_df.columns]
            price_df.to_csv(file_path)
        self.price = price_df
        return price_df

    def group_split(self, df, start_year):
        '''
        Split the data of each group into train(first 5 years) and test(last 1 year)
        '''
        length = self.group_test_len + self.group_train_len
        end_year = start_year + length - 1

        ix = df.index
        start = ix[ix.year == start_year][0]
        end = ix[ix.year == end_year][-1]
        data = df.loc[start:end]
        return data

    def get_groups(self, group_years, df):
        '''
        Get the train+test data for each group

        Return:
            dict
        '''
        data_groups = {
            year: self.group_split(
                df, year) for year in group_years}
        return data_groups

    def create_labels(self, df, col_name='close', window_size=11):
        """
        Data is labeled as per the logic in research paper
        Label code : BUY => 1, SELL => 0, HOLD => 2

        params :
            df => Dataframe with data
            col_name => name of column which should be used to determine strategy

        returns : numpy array with integer codes for labels with
                  size = total-(window_size)+1
        """

        row_counter = 0
        total_rows = len(df)
        labels = np.zeros(total_rows)
        labels[:] = np.nan
        print("Calculating labels")
        pbar = tqdm(total=total_rows)

        while row_counter < total_rows:
            if row_counter >= window_size - 1:
                window_begin = row_counter - (window_size - 1)
                window_end = row_counter
                window_middle = (window_begin + window_end) / 2

                min_ = np.inf
                min_index = -1
                max_ = -np.inf
                max_index = -1
                for i in range(window_begin, window_end + 1):
                    price = df.iloc[i][col_name]
                    if price < min_:
                        min_ = price
                        min_index = i
                    if price > max_:
                        max_ = price
                        max_index = i

                if max_index == window_middle:
                    labels[row_counter] = 0
                elif min_index == window_middle:
                    labels[row_counter] = 1
                else:
                    labels[row_counter] = 2

            row_counter = row_counter + 1
            pbar.update(1)

        pbar.close()
        return labels

    def cal_tech(self, df, col_name, intervals, whole_df=False):

        year = str(df.index[0].year)
        if not whole_df:
            file_name = self.ticker + '_' + year + '_' + 'Tech.csv'
            file_path = self.ticker_path + os.sep + self.ticker + '_' + year
            if not os.path.exists(file_path):
                os.makedirs(file_path)
        else:
            file_name = self.ticker + '_All_Tech.csv'
            file_path = self.ticker_path
            if os.path.exists(file_path + os.sep + file_name):
                self.df_Tech = pd.read_csv(
                    file_path + os.sep + file_name, index_col=0, parse_dates=[0])
                return self.df_Tech

        get_RSI_smooth(df, col_name, intervals)  # momentum
        get_williamR(df, col_name, intervals)  # momentum
        get_mfi(df, intervals)  # momentum
        get_ROC(df, col_name, intervals)  # momentum
        get_CMF(df, col_name, intervals)  # momentum, volume EMA
        get_CMO(df, col_name, intervals)  # momentum
        get_SMA(df, col_name, intervals)
        get_SMA(df, 'open', intervals)
        get_EMA(df, col_name, intervals)
        get_WMA(df, col_name, intervals)
        get_HMA(df, col_name, intervals)
        get_TRIX(df, col_name, intervals)  # trend
        get_CCI(df, col_name, intervals)  # trend
        get_DPO(df, col_name, intervals)  # Trend oscillator
        get_kst(df, col_name, intervals)  # Trend
        get_DMI(df, col_name, intervals)  # trend
        get_BB_MAV(df, col_name, intervals)  # volatility
        get_force_index(df, intervals)  # volume
        get_EOM(df, col_name, intervals)  # volume momentum

        print('\n{} features of {} rows are created totally'.format(
            len(df.columns) - 5, len(df)))

        df.to_csv(file_path + os.sep + file_name)
        self.df_Tech = df
        return df

    def feature_selection(self, x_train, y_train, top_k,
                          num_features, seed=None):
        '''
        Get the best features from the technical indicators calculated above
        Select the common features based on the f_classif and mutual_info_classif methods

        params:
            num_features: the number of features we need at last (Need to be n**2)
            top_k: the number of features selected from each selectKBest method
        '''
        mm_scaler = MinMaxScaler(feature_range=(0, 1))  # or StandardScaler?
        x = mm_scaler.fit_transform(x_train.values)
        y = y_train.values

        select_k_best = SelectKBest(f_classif, k=top_k)
        select_k_best.fit(x, y)
        # Get the integer index of the features selected
        k_idx1 = select_k_best.get_support(indices=True)

        select_k_best2 = SelectKBest(mutual_info_classif, k=top_k)
        select_k_best2.fit(x, y)
        k_idx2 = select_k_best2.get_support(indices=True)

        common = list(set(k_idx1).intersection(set(k_idx2)))

        if len(common) < num_features:
            raise Exception(
                'number of common features found {} < {} required features. Increase "topK"'.format(len(common),
                                                                                                    num_features))
        if seed is not None:
            np.random.seed(seed)
        feat_idx = np.random.choice(common, num_features, replace=False)
        return sorted(feat_idx)

    def reshape_as_image(self, x, img_width, img_height):
        x_temp = np.zeros((len(x), img_height, img_width))
        for i in range(x.shape[0]):
            x_temp[i] = np.reshape(x[i], (img_height, img_width))
        return x_temp

    def get_sample_weights(self, y):
        """
        calculate the sample weights based on class weights. Used for models with
        imbalanced data and one hot encoding prediction.

        params:
            y: class labels as integers
        """

        y = y.astype(int)  # compute_class_weight needs int labels
        class_weights = compute_class_weight('balanced', np.unique(y), y)

        print("\nreal class weights are {}".format(class_weights), np.unique(y))
        print("value_counts", np.unique(y, return_counts=True))
        sample_weights = y.copy().astype(float)
        for i in np.unique(y):
            sample_weights[sample_weights == i] = class_weights[i]

        return sample_weights

    def train_test_split(self, df):
        '''
        Split data into train and test

        Return:
            dataframe
        '''
        ix = df.index
        start_year = ix[0].year
        end_year = start_year + self.group_train_len + self.group_test_len - 1

        train_start = ix[ix.year == start_year][0]
        train_end = ix[ix.year == end_year - 1][-1]
        test_start = ix[ix.year == end_year][0]
        test_end = ix[ix.year == end_year][-1]

        train_data = df.loc[train_start:train_end]
        test_data = df.loc[test_start:test_end]

        x_train_df = train_data.iloc[:, :-1]
        y_train_df = train_data.iloc[:, -1]
        x_test_df = test_data.iloc[:, :-1]
        y_test_df = test_data.iloc[:, -1]

        return x_train_df, x_test_df, y_train_df, y_test_df

    def Process_Data(self, df, top_k, dim=15,
                     split_rdm_state=1, cv_size=0.2, save=True):
        '''
        Data processing; Produce available data that is input into model for one group
        Whole procedures are as follows:
            Creating labels
            Feature selection
            Split train and cv
            Class weights calculation
            One-hot-encoder

        Parameters:
            df: the dataframe containing the calculated technical indicators
            top_k: number of top features in feature selection
            dim: the dimension of the image size in CNN
        Return:
            x_train, x_cv, y_train, y_cv, sample_weights
        '''
        year = str(df.index[0].year)
        file_path = self.ticker_path + os.sep + self.ticker + '_' + year
        if not os.path.exists(file_path):
            os.makedirs(file_path)

        # Create Labels
        df['labels'] = self.create_labels(df, 'close')
        df.dropna(inplace=True)

        # Split train, test, cv
        x_main_df, x_test_df, y_main_df, y_test_df = self.train_test_split(df)
        x_train_df, x_cv_df, y_train_df, y_cv_df = train_test_split(
            x_main_df, y_main_df, train_size=1 - cv_size, test_size=cv_size, random_state=split_rdm_state, shuffle=True, stratify=y_main_df)

        # Calculate sample weights
        train_class_weights = self.get_sample_weights(y_train_df.values)

        # feature selection
        num_features = dim**2
        feat_idx = self.feature_selection(
            x_train_df, y_train_df, top_k, num_features)
        feature_list = list(df.iloc[:, :-1].columns)
        feat_name = [feature_list[i] for i in feat_idx]
        print('\nSelected Features:', feat_name)

        x_train = x_train_df.iloc[:, feat_idx].values
        x_cv = x_cv_df.iloc[:, feat_idx].values
        x_test = x_test_df.iloc[:, feat_idx].values

        # Standarlize
        # mm_scaler = MinMaxScaler(feature_range=(0, 1))  # or StandardScaler?
        mm_scaler = StandardScaler()
        x_train = mm_scaler.fit_transform(x_train)
        x_cv = mm_scaler.transform(x_cv)
        x_test = mm_scaler.transform(x_test)

        assert dim == int(
            np.sqrt(
                x_train.shape[1])), 'Dimension of data has problem'
        x_train = self.reshape_as_image(x_train, dim, dim)
        x_cv = self.reshape_as_image(x_cv, dim, dim)
        x_test = self.reshape_as_image(x_test, dim, dim)

        x_train = np.stack((x_train,) * 3, axis=-1)  # m*height*width*3
        x_cv = np.stack((x_cv,) * 3, axis=-1)
        x_test = np.stack((x_test,) * 3, axis=-1)

        one_hot_enc = OneHotEncoder(sparse=False, categories='auto')
        y_train = one_hot_enc.fit(
            y_train_df.values.reshape(-1, 1)).transform(y_train_df.values.reshape(-1, 1))
        y_cv = one_hot_enc.fit(
            y_cv_df.values.reshape(-1, 1)).transform(y_cv_df.values.reshape(-1, 1))
        y_test = one_hot_enc.fit(
            y_test_df.values.reshape(-1, 1)).transform(y_test_df.values.reshape(-1, 1))

        if save:
            np.save(file=file_path + os.sep + "x_train.npy", arr=x_train)
            np.save(file=file_path + os.sep + "x_cv.npy", arr=x_cv)
            np.save(file=file_path + os.sep + "x_test.npy", arr=x_test)
            np.save(file=file_path + os.sep + "y_train.npy", arr=y_train)
            np.save(file=file_path + os.sep + "y_cv.npy", arr=y_cv)
            np.save(file=file_path + os.sep + "y_test.npy", arr=y_test)
            np.save(
                file=file_path +
                os.sep +
                "train_class_weights.npy",
                arr=train_class_weights)

        return x_train, x_cv, x_test, y_train, y_cv, y_test, train_class_weights

    def load_np_data(self, year):
        path = self.ticker_path + os.sep + \
            self.ticker + '_' + str(year) + os.sep
        x_train = np.load(file=path + "x_train.npy")
        x_cv = np.load(file=path + "x_cv.npy")
        x_test = np.load(file=path + "x_test.npy")
        y_train = np.load(file=path + "y_train.npy")
        y_cv = np.load(file=path + "y_cv.npy")
        y_test = np.load(file=path + "y_test.npy")
        train_class_weights = np.load(file=path + "train_class_weights.npy")
        return x_train, x_cv, x_test, y_train, y_cv, y_test, train_class_weights


if __name__ == '__main__':
    Data_gen = Data('AAPL')

