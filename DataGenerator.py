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

    def label_to_numeric(self, row):
        return self.label_map[row]

    def feed_csv(self, columns, file, ADL, SUBJECT_ID, TRIAL_NO):
        df = pd.read_csv(file, skiprows=16, names=columns)
        df['activity'] = ADL
        df['subject_id'] = SUBJECT_ID
        df['trial_id'] = TRIAL_NO.split('.')[0]
        # train test split based on ratio
        if self.istrain:
            df = df.iloc[0:int(df.shape[0] * (1 - self.ratio)) - 1]
        else:
            df = df.iloc[(df.shape[0] - int(df.shape[0] * self.ratio) - 1):df.shape[0]]
        self._all_data = self._all_data.append(df, ignore_index=True, sort=True)

    def __init__(self, dataset_pattern_path, train_for='acc', extract_data_size=30, istrain=False, ratio=0.3):
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

    def get_data_files(self):
        return self._datafiles

    def get_batch(self, batchsize, start_list=None):

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
            label_y.append(to_categorical(y.values))

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
        y = np.array(to_categorical(y.values))
        return x, y

    def on_epoch_begin(self, epoch, logs={}):
        # pass all the subjects one by with all the activities one by one
        if epoch < 8:
            self._target_subject_id = '1'
        elif 8 <= epoch < 14:
            self._target_subject_id = '2'
        elif 14 <= epoch < 20:
            self._target_subject_id = '3'
        elif 20 <= epoch < 26:
            self._target_subject_id = '4'
        elif 26 <= epoch < 32:
            self._target_subject_id = '5'


generator = MobiFallGenerator('./dataset/*/*/*/*/*.txt')


print(generator.get_test_data())
