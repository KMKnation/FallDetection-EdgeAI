import pandas as pd
import numpy as np
import glob
import random

from tensorflow.python.keras.utils import to_categorical


class MobiFallGenerator(object):
    acc_columns = ['timestamp', 'x', 'y', 'z(m/s^2)']
    ori_columns = ['timestamp', 'Azimuth', 'Pitch', 'Roll']
    gyro_columns = ['timestamp', 'g_x', 'g_y', 'z(rad/s)']
    common_cols = ['activity', 'subject_id', 'trial_id', 'timestamp']
    label_map = {
        'STD': 0,
        'WAL': 1,
        'JOG': 2,
        'JUM': 3,
        'STU': 4,
        'STN': 5,
        'SCH': 6,
        'CSI': 7,
        'CSO': 8,
        'FOL': 9,
        'FKL': 10,
        'BSC': 11,
        'SDL': 12
    }

    def __init__(self, dataset_pattern_path, train_for='acc', extract_data_size=1, istrain=False, ratio=0.3):
        self._datafiles = glob.glob(dataset_pattern_path)
        self._extract_data_size = extract_data_size
        self.istrain = istrain
        self.ratio = ratio
        self._train_for = train_for
        self._target_subject_id = '1'

        if self._train_for == 'acc':
            self._all_data = pd.DataFrame(columns=self.acc_columns)
            self.columns = self.acc_columns
        elif self._train_for == 'ori':
            self._all_data = pd.DataFrame(columns=self.ori_columns)
            self.columns = self.ori_columns
        elif self._train_for == 'gyro':
            self._all_data = pd.DataFrame(columns=self.gyro_columns)
            self.columns = self.gyro_columns

        self.cols_to_train = [column for column in self.columns if column not in self.common_cols]
        self.cols_to_train.append('timestamp')
        print("TRAINING FOR " + str(self.cols_to_train))

        for file in self._datafiles:
            ADL, SENSOR_CODE, SUBJECT_ID, TRIAL_NO = file.split('/')[-1].split('_')

            if SENSOR_CODE != self._train_for:
                continue

            if SENSOR_CODE == 'acc' and self._train_for == 'acc':
                self.feed_csv(self.acc_columns, file, ADL, SUBJECT_ID, TRIAL_NO)
            elif SENSOR_CODE == 'ori' and self._train_for == 'ori':
                self.feed_csv(self.ori_columns, file, ADL, SUBJECT_ID, TRIAL_NO)
            elif SENSOR_CODE == 'gyro' and self._train_for == 'gyro':
                self.feed_csv(self.gyro_columns, file, ADL, SUBJECT_ID, TRIAL_NO)

    def label_to_numeric(self, row):
        return self.label_map[row]

    def feed_csv(self, columns, file, ADL, SUBJECT_ID, TRIAL_NO):
        df = pd.read_csv(file, skiprows=16, names=columns, nrows=120)
        df['activity'] = ADL
        df['subject_id'] = SUBJECT_ID
        df['trial_id'] = TRIAL_NO.split('.')[0]
        # train test split based on ratio
        if self.istrain:
            df = df.iloc[0:int(df.shape[0] * (1 - self.ratio)) - 1]
        else:
            df = df.iloc[(df.shape[0] - int(df.shape[0] * self.ratio) - 1):df.shape[0]]
        self._all_data = self._all_data.append(df, ignore_index=True, sort=True)

    def get_data_files(self):
        return self._datafiles

    #
    # def get_batch(self, batchsize=1, start_list=None):
    #     '''
    #         LSTM (and GRU) layers require 3 dimensional inputs:
    #         a batch size, a number of time steps, and a number of features.
    #     :param batchsize:
    #     :param start_list:
    #     :return:
    #     '''
    #
    #     target_df = self._all_data[(self._all_data['subject_id'] == self._target_subject_id)].sort_values(
    #         by=['activity', 'subject_id', 'trial_id', 'timestamp'], ascending=[True, True, True, True])
    #     data_size = target_df.shape[0]
    #
    #     if start_list is None:
    #         start_pos = [random.randint(1, data_size - self._extract_data_size) for _ in range(data_size)]
    #     else:
    #         if len(start_list) != batchsize:
    #             print('batchisze = ', batchsize)
    #             print('start_list length = ', len(start_list))
    #             raise KeyError('batchsize is no equal to start_list length!')
    #         start_pos = start_list
    #
    #     col_indexes = [target_df.columns.get_loc(column) for column in self.cols_to_train]
    #     col_indexes.sort()
    #
    #     i = 0
    #     y = target_df.iloc[start_pos[i]:start_pos[i] + self._extract_data_size,
    #         target_df.columns.get_loc("activity")]
    #     y = y.apply(lambda row: self.label_to_numeric(row))
    #
    #     yield target_df.iloc[start_pos[i]:start_pos[i] + self._extract_data_size, col_indexes].values, y.values

    def get_batch(self, batchsize=1, start_list=None):
        '''
            LSTM (and GRU) layers require 3 dimensional inputs:
            a batch size, a number of time steps, and a number of features.
        '''

        while True:

            # keep in mind that whatever timestamps you pass in one bundle of x
            # the target label for that timestamps should be same

            # there is 500 timestasmps minumum for each activity

            target_df = self._all_data[(self._all_data['subject_id'] == self._target_subject_id)].sort_values(
                by=['activity', 'subject_id', 'trial_id', 'timestamp'], ascending=[True, True, True, True])
            # choose random activity of that subject
            activities = target_df['activity'].unique()
            target_activity = activities[random.randint(0, len(activities) - 1)]
            target_df = target_df[target_df['activity'] == target_activity]

            # print(target_df.head())
            data_size = target_df.shape[0]
            # print(data_size)

            if self._extract_data_size > data_size:
                raise Exception("Not enough data")

            if start_list is None:
                start_pos = [random.randint(1, data_size - self._extract_data_size) for _ in range(data_size)]
            else:
                if len(start_list) != batchsize:
                    print('batchisze = ', batchsize)
                    print('start_list length = ', len(start_list))
                    raise KeyError('batchsize is no equal to start_list length!')
                start_pos = start_list

            train_x = []
            train_y = []
            col_indexes = [target_df.columns.get_loc(column) for column in self.cols_to_train]
            col_indexes.sort()

            if data_size < batchsize:
                print(np.array(train_x).shape)
                print(np.array(train_y).shape)

                '''
                correct shape
                '''
                x_train = np.random.random((batchsize, self._extract_data_size, self.get_features_count()))
                y_train = np.random.random((batchsize, self.get_total_categories()))

                print(x_train.shape)
                print(y_train.shape)
                raise Exception("Batch size is so high then the acctual data size")

            for i in range(batchsize):
                train_x.append(target_df.iloc[start_pos[i]:start_pos[i] + self._extract_data_size, col_indexes].values)
                train_y.append(
                    to_categorical(self.label_to_numeric(target_activity), num_classes=self.get_total_categories()))

            # yield x_train, y_train

            yield np.array(train_x), np.array(train_y)

    def get_batch_old(self, batchsize=1, start_list=None):
        '''
            LSTM (and GRU) layers require 3 dimensional inputs:
            a batch size, a number of time steps, and a number of features.
        :param batchsize:
        :param start_list:
        :return:
        '''

        target_df = self._all_data[(self._all_data['subject_id'] == self._target_subject_id)].sort_values(
            by=['activity', 'subject_id', 'trial_id', 'timestamp'], ascending=[True, True, True, True])
        data_size = target_df.shape[0]

        if start_list is None:
            start_pos = [random.randint(1, data_size - self._extract_data_size) for _ in range(data_size)]
        else:
            if len(start_list) != batchsize:
                print('batchisze = ', batchsize)
                print('start_list length = ', len(start_list))
                raise KeyError('batchsize is no equal to start_list length!')
            start_pos = start_list

        train_x = []
        label_y = []
        col_indexes = [target_df.columns.get_loc(column) for column in self.cols_to_train]
        col_indexes.sort()

        for i in range(batchsize):
            train_x.append(target_df.iloc[start_pos[i]:start_pos[i] + self._extract_data_size, col_indexes].values)
            y = target_df.iloc[start_pos[i]:start_pos[i] + self._extract_data_size,
                target_df.columns.get_loc("activity")]
            y = y.apply(lambda row: self.label_to_numeric(row))

            label_y.append(to_categorical(y.iloc[0], num_classes=self.get_total_categories()))

        return np.array(train_x), np.array(label_y)

    def get_test_data(self):
        """
        x shape = [datasize, 3]
        y shape = [datasize ,1]
        :return:
        """

        target_df = self._all_data[(self._all_data['subject_id'] == self._target_subject_id)].sort_values(
            by=self.common_cols, ascending=[True, True, True, True])

        col_indexes = [target_df.columns.get_loc(column) for column in self.cols_to_train]
        col_indexes.sort()
        x = np.array(target_df.iloc[0:10, col_indexes].values)
        y = target_df.iloc[0:10, target_df.columns.get_loc("activity")]
        y = y.apply(lambda row: self.label_to_numeric(row))
        y = np.array(y.values)
        return x, to_categorical(y)

    def on_epoch_begin(self, epoch, logs={}):
        # pass all the subjects one by with all the activities one by one
        if epoch < 8:
            self._target_subject_id = '1'
        elif 8 \
                <= epoch < 14:
            self._target_subject_id = '2'
        elif 14 <= epoch < 20:
            self._target_subject_id = '3'
        elif 20 <= epoch < 26:
            self._target_subject_id = '4'
        elif 26 <= epoch < 32:
            self._target_subject_id = '5'

    def get_observations_per_epoch(self):
        return self._extract_data_size

    def get_features_count(self):
        return len(self.cols_to_train)

    def get_total_categories(self):
        return len(self.label_map.keys())

# generator = MobiFallGenerator('./dataset/*/*/*/*/*.txt', istrain=False)
# x, y = generator.get_batch(100 * 10)
# print(x.shape)
#
# print(y.shape)

#
# # #
# # print(generator.get_test_data())
# # print(generator.get_observations_per_epoch())
# # print(generator.get_features_count())
# # print(generator.get_total_categories())
