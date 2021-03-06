import pandas as pd
import numpy as np
import glob
import random
import gc
from tensorflow import keras
from tensorflow.python.keras.utils.np_utils import to_categorical


class MobiDataGenerator(keras.callbacks.Callback):
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

    subjects = [i for i in range(1, 22)]
    subjects.append(29)
    subjects.append(30)
    subjects.append(31)

    def __init__(self, dataset_pattern_path, train_for='acc',
                 batch_size=64,  # steps_per_epoch * epochs
                 extract_data_size=2,
                 istrain=False,
                 ratio=0.3):
        self._datafiles = glob.glob(dataset_pattern_path)
        self._extract_data_size = extract_data_size
        self.istrain = istrain
        self.ratio = ratio
        self._train_for = train_for
        self._target_subject_id = '2'
        self.batch_size = batch_size

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

        print('TRAINING DATA SHAPE {}'+str(self._all_data.shape))

        print('DATA LOADED !!')

    def label_to_numeric(self, activity):
        try:
            return self.label_map[activity]
        except Exception as err:
            print(err)
            return int(activity)

    def feed_csv(self, columns, file, ADL, SUBJECT_ID, TRIAL_NO):
        df = pd.read_csv(file, skiprows=16, names=columns)
        df['activity'] = ADL
        df['subject_id'] = SUBJECT_ID
        df['trial_id'] = TRIAL_NO.split('.')[0]
        # train test split based on ratio

        self._all_data = self._all_data.append(df.iloc[0:], ignore_index=True, sort=True)

    def get_data_files(self):
        return self._datafiles

    def prepare_data(self, activities, main_df, batchsize):
        # target_activity = activities[random.randint(0, len(activities) - 1)]
        # print(' ACTIVITIES {}'.format(str(main_df['activity'].unique())))

        target_df = main_df.copy()
        # target_df = main_df[main_df['activity'] == target_activity]
        target_df['activity'] = target_df['activity'].apply(lambda activity: self.label_to_numeric(activity))

        # print(target_df.head())
        data_size = target_df.shape[0]
        # print(data_size)

        if self._extract_data_size > data_size:
            raise Exception("Not enough data You are training for more timesteps then actual data")

        start_pos = [random.randint(1, data_size - self._extract_data_size) for _ in range(data_size)]

        train_x = []
        train_y = []
        col_indexes = [target_df.columns.get_loc(column) for column in self.cols_to_train]
        col_indexes.sort()

        for i in range(batchsize):
            if len(train_y) == batchsize:
                break

            if i == data_size:
                break
            train_x.append(target_df.iloc[start_pos[i]:start_pos[i] + self._extract_data_size, col_indexes].values)

            # get max vote for this region of timestamp
            labels = target_df.iloc[start_pos[i]:start_pos[i] + self._extract_data_size,
                     target_df.columns.get_loc('activity')].values
            labels = sorted(labels, key=list(labels).count)
            train_y.append(to_categorical(labels[-1], num_classes=self.get_total_categories()))

        while len(train_y) < batchsize:

            if len(train_y) == batchsize:
                break

            # print('some how less data for one activity for generating more for same subject')
            x, y = self.prepare_data(activities, main_df, batchsize - len(train_y))
            for i in range(len(x)):
                train_x.append(x[i])
                train_y.append(y[i])

        return train_x, train_y

    def get_batch(self, batchsize=1, isTrain=True):
        '''
            LSTM (and GRU) layers require 3 dimensional inputs:
            a batch size, a number of time steps, and a number of features.
        '''

        # keep in mind that whatever timestamps you pass in one bundle of x
        # the target label for that timestamps should be same


        # there is 500 timestasmps minumum for each activity
        if isTrain:
            target_df = self._all_data[(self._all_data['subject_id'] == self._target_subject_id)].sort_values(
                by=['timestamp', 'subject_id', 'trial_id'], ascending=[True, True, True])
        else:
            target_df = self._test_data[(self._test_data['subject_id'] == self._target_subject_id)].sort_values(
                by=['timestamp', 'subject_id', 'trial_id'], ascending=[True, True, True])

        # print(target_df.head())
        # print(self.cols_to_train)
        # exit(0)

        # removing timestamps after ordering
        target_df = target_df.drop(['timestamp'], axis=1)
        if 'timestamp' in self.cols_to_train:
            self.cols_to_train.remove('timestamp')

        # total_size = target_df.shape[0]

        # if total_size < batchsize:
        #     '''
        #     correct shape
        #     '''
        #     x_train = np.random.random((batchsize, self._extract_data_size, self.get_features_count()))
        #     y_train = np.random.random((batchsize, self.get_total_categories()))
        #
        #     print(x_train.shape)
        #     print(y_train.shape)
        #     print("Total Size {}".format(total_size))
        #     raise Exception("Batch size is so high then the acctual data size")

        # choose random activity of that subject
        activities = target_df['activity'].unique()

        train_x, train_y = self.prepare_data(activities, target_df, batchsize)

        return (np.array(train_x), np.array(train_y))

    def next_train(self):
        while 1:
            ret = self.get_batch(self.batch_size, True)
            yield ret

    def next_val(self):
        while 1:
            ret = self.get_batch(self.batch_size, False)
            yield ret

    def get_test_data(self, subject_id=None, activity = None):
        """
        x shape = [datasize, 3]
        y shape = [data-size ,1]
        :return:
        """
        target_df = self._all_data[self._all_data['subject_id'] == str(subject_id)]
        if activity != None:
           target_df = target_df[target_df['activity'] == activity]
        target_df = target_df.sort_values(
            by=['timestamp', 'subject_id', 'trial_id'], ascending=[True, True, True])
        train_x = []
        train_y = []

        # print(target_df.head())
        data_size = target_df.shape[0]
        # print(data_size)

        if self._extract_data_size > data_size:
            raise Exception("Not enough data")

        start_pos = [i for i in range(0, data_size - self._extract_data_size, self._extract_data_size )]
        self.cols_to_train.remove('timestamp')
        col_indexes = [target_df.columns.get_loc(column) for column in self.cols_to_train]
        activity_col_index = target_df.columns.get_loc('activity')

        col_indexes.sort()

        for i in range(len(start_pos)):
            if len(train_y) == len(start_pos):
                break

            if i == data_size:
                break

            train_x.append(target_df.iloc[start_pos[i]:start_pos[i] + self._extract_data_size, col_indexes].values)
            train_y.append(
                target_df.iloc[start_pos[i]:start_pos[i] + self._extract_data_size, activity_col_index].values)

        while len(train_y) < len(start_pos):

            if len(train_y) == len(start_pos):
                break

            # print('some how less data for one activity for generating more for same subject')
            x, y,_ = self.get_test_data(subject_id=subject_id, activity=activity)
            for i in range(len(x)):
                train_x.append(x[i])
                train_y.append(y[i])

        return train_x, train_y, self.cols_to_train



    def on_batch_begin(self, batch, logs=None):

        self._target_subject_id = str(self.subjects[random.randint(0, len(self.subjects) - 1)])

        # # pass all the subjects one by with all the activities one by one
        # if epoch < 3:
        #     self._target_subject_id = '2'
        # elif 3 <= epoch < 5:
        #     self._target_subject_id = '1'
        # elif 5 <= epoch < 7:
        #     self._target_subject_id = '3'
        # elif 7 <= epoch < 9:
        #     self._target_subject_id = '4'
        # elif 9 <= epoch < 11:
        #     self._target_subject_id = '5'
        # elif 11 <= epoch < 13:
        #     self._target_subject_id = '6'
        # elif 13 <= epoch < 15:
        #     self._target_subject_id = '7'
        # elif 15 <= epoch < 17:
        #     self._target_subject_id = '9'
        # elif 17 <= epoch < 19:
        #     self._target_subject_id = '10'
        # elif 19 <= epoch < 50:
        #     self._target_subject_id = str(self.subjects[random.randint(0, len(self.subjects) - 1)])

        # elif 50 <= epoch < 52:
        #     self._target_subject_id = '12'
        # elif 52 <= epoch < 54:
        #     self._target_subject_id = '13'
        # elif 54 <= epoch < 60:
        #     self._target_subject_id = '14'
        # elif 60 <= epoch < 62:
        #     self._target_subject_id = '15'
        # elif 62 <= epoch < 64:
        #     self._target_subject_id = '16'
        # elif epoch >= 64:

        print(" WE ARE TRAINING FOR SUBJECT ID = {}".format(self._target_subject_id))

        return super().on_batch_begin(batch, logs)

    def on_epoch_begin(self, epoch, logs={}):
        print('')

    def get_observations_per_epoch(self):
        return self._extract_data_size

    def get_features_count(self):
        # removing timestamp columns
        return len(self.cols_to_train) - 1

    def get_total_categories(self):
        return len(self.label_map.keys())

class MobiFallGenerator(keras.callbacks.Callback):
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

    subjects = [i for i in range(1, 22)]
    subjects.append(29)
    subjects.append(30)
    subjects.append(31)

    def __init__(self, dataset_pattern_path, train_for='acc',
                 batch_size=64,  # steps_per_epoch * epochs
                 extract_data_size=2,
                 istrain=False,
                 ratio=0.3):
        self._datafiles = glob.glob(dataset_pattern_path)
        self._extract_data_size = extract_data_size
        self.istrain = istrain
        self.ratio = ratio
        self._train_for = train_for
        self._target_subject_id = '2'
        self.batch_size = batch_size

        if self._train_for == 'acc':
            self._all_data = pd.DataFrame(columns=self.acc_columns)
            self._test_data = pd.DataFrame(columns=self.acc_columns)
            self.columns = self.acc_columns
        elif self._train_for == 'ori':
            self._all_data = pd.DataFrame(columns=self.ori_columns)
            self._test_data = pd.DataFrame(columns=self.ori_columns)
            self.columns = self.ori_columns
        elif self._train_for == 'gyro':
            self._all_data = pd.DataFrame(columns=self.gyro_columns)
            self._test_data = pd.DataFrame(columns=self.gyro_columns)
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

        print('TRAINING DATA SHAPE {}'+str(self._all_data.shape))
        print('TESTING DATA SHAPE {}'+str(self._test_data.shape))

        print('DATA LOADED !!')

    def label_to_numeric(self, activity):
        try:
            return self.label_map[activity]
        except Exception as err:
            print(err)
            return int(activity)

    def feed_csv(self, columns, file, ADL, SUBJECT_ID, TRIAL_NO):
        df = pd.read_csv(file, skiprows=16, names=columns)
        df['activity'] = ADL
        df['subject_id'] = SUBJECT_ID
        df['trial_id'] = TRIAL_NO.split('.')[0]
        # train test split based on ratio

        trainig_upto = int(df.shape[0] * (1 - self.ratio)) - 1
        self._all_data = self._all_data.append(df.iloc[0:trainig_upto], ignore_index=True, sort=True)
        testing_from = (df.shape[0] - int(df.shape[0] * self.ratio) - 1)
        self._test_data = self._test_data.append(df.iloc[testing_from:df.shape[0]], ignore_index=True, sort=True)

    def get_data_files(self):
        return self._datafiles

    def prepare_data(self, activities, main_df, batchsize):
        # target_activity = activities[random.randint(0, len(activities) - 1)]
        # print(' ACTIVITIES {}'.format(str(main_df['activity'].unique())))

        target_df = main_df.copy()
        # target_df = main_df[main_df['activity'] == target_activity]
        target_df['activity'] = target_df['activity'].apply(lambda activity: self.label_to_numeric(activity))

        # print(target_df.head())
        data_size = target_df.shape[0]
        # print(data_size)

        if self._extract_data_size > data_size:
            raise Exception("Not enough data You are training for more timesteps then actual data")

        start_pos = [random.randint(1, data_size - self._extract_data_size) for _ in range(data_size)]

        train_x = []
        train_y = []
        col_indexes = [target_df.columns.get_loc(column) for column in self.cols_to_train]
        col_indexes.sort()

        for i in range(batchsize):
            if len(train_y) == batchsize:
                break

            if i == data_size:
                break
            train_x.append(target_df.iloc[start_pos[i]:start_pos[i] + self._extract_data_size, col_indexes].values)

            # get max vote for this region of timestamp
            labels = target_df.iloc[start_pos[i]:start_pos[i] + self._extract_data_size,
                     target_df.columns.get_loc('activity')].values
            labels = sorted(labels, key=list(labels).count)
            train_y.append(to_categorical(labels[-1], num_classes=self.get_total_categories()))

        while len(train_y) < batchsize:

            if len(train_y) == batchsize:
                break

            # print('some how less data for one activity for generating more for same subject')
            x, y = self.prepare_data(activities, main_df, batchsize - len(train_y))
            for i in range(len(x)):
                train_x.append(x[i])
                train_y.append(y[i])

        return train_x, train_y

    def get_batch(self, batchsize=1, isTrain=True):
        '''
            LSTM (and GRU) layers require 3 dimensional inputs:
            a batch size, a number of time steps, and a number of features.
        '''

        # keep in mind that whatever timestamps you pass in one bundle of x
        # the target label for that timestamps should be same


        # there is 500 timestasmps minumum for each activity
        if isTrain:
            target_df = self._all_data[(self._all_data['subject_id'] == self._target_subject_id)].sort_values(
                by=['timestamp', 'subject_id', 'trial_id'], ascending=[True, True, True])
        else:
            target_df = self._test_data[(self._test_data['subject_id'] == self._target_subject_id)].sort_values(
                by=['timestamp', 'subject_id', 'trial_id'], ascending=[True, True, True])

        # print(target_df.head())
        # print(self.cols_to_train)
        # exit(0)

        # removing timestamps after ordering
        target_df = target_df.drop(['timestamp'], axis=1)
        if 'timestamp' in self.cols_to_train:
            self.cols_to_train.remove('timestamp')

        # total_size = target_df.shape[0]

        # if total_size < batchsize:
        #     '''
        #     correct shape
        #     '''
        #     x_train = np.random.random((batchsize, self._extract_data_size, self.get_features_count()))
        #     y_train = np.random.random((batchsize, self.get_total_categories()))
        #
        #     print(x_train.shape)
        #     print(y_train.shape)
        #     print("Total Size {}".format(total_size))
        #     raise Exception("Batch size is so high then the acctual data size")

        # choose random activity of that subject
        activities = target_df['activity'].unique()

        train_x, train_y = self.prepare_data(activities, target_df, batchsize)

        return (np.array(train_x), np.array(train_y))

    def next_train(self):
        while 1:
            ret = self.get_batch(self.batch_size, True)
            yield ret

    def next_val(self):
        while 1:
            ret = self.get_batch(self.batch_size, False)
            yield ret

    def get_test_data(self, subject_id=None, activity = None):
        """
        x shape = [datasize, 3]
        y shape = [data-size ,1]
        :return:
        """
        target_df = self._all_data[self._all_data['subject_id'] == str(subject_id)]
        if activity != None:
           target_df = target_df[target_df['activity'] == activity]
        target_df = target_df.sort_values(
            by=['timestamp', 'subject_id', 'trial_id'], ascending=[True, True, True])
        train_x = []
        train_y = []

        # print(target_df.head())
        data_size = target_df.shape[0]
        # print(data_size)

        if self._extract_data_size > data_size:
            raise Exception("Not enough data")

        start_pos = [i for i in range(0, data_size - self._extract_data_size, self._extract_data_size )]
        self.cols_to_train.remove('timestamp')
        col_indexes = [target_df.columns.get_loc(column) for column in self.cols_to_train]
        activity_col_index = target_df.columns.get_loc('activity')

        col_indexes.sort()

        for i in range(len(start_pos)):
            if len(train_y) == len(start_pos):
                break

            if i == data_size:
                break

            train_x.append(target_df.iloc[start_pos[i]:start_pos[i] + self._extract_data_size, col_indexes].values)
            train_y.append(
                target_df.iloc[start_pos[i]:start_pos[i] + self._extract_data_size, activity_col_index].values)

        while len(train_y) < len(start_pos):

            if len(train_y) == len(start_pos):
                break

            # print('some how less data for one activity for generating more for same subject')
            x, y,_ = self.get_test_data(subject_id=subject_id, activity=activity)
            for i in range(len(x)):
                train_x.append(x[i])
                train_y.append(y[i])

        return train_x, train_y, self.cols_to_train



    def on_batch_begin(self, batch, logs=None):

        self._target_subject_id = str(self.subjects[random.randint(0, len(self.subjects) - 1)])

        # # pass all the subjects one by with all the activities one by one
        # if epoch < 3:
        #     self._target_subject_id = '2'
        # elif 3 <= epoch < 5:
        #     self._target_subject_id = '1'
        # elif 5 <= epoch < 7:
        #     self._target_subject_id = '3'
        # elif 7 <= epoch < 9:
        #     self._target_subject_id = '4'
        # elif 9 <= epoch < 11:
        #     self._target_subject_id = '5'
        # elif 11 <= epoch < 13:
        #     self._target_subject_id = '6'
        # elif 13 <= epoch < 15:
        #     self._target_subject_id = '7'
        # elif 15 <= epoch < 17:
        #     self._target_subject_id = '9'
        # elif 17 <= epoch < 19:
        #     self._target_subject_id = '10'
        # elif 19 <= epoch < 50:
        #     self._target_subject_id = str(self.subjects[random.randint(0, len(self.subjects) - 1)])

        # elif 50 <= epoch < 52:
        #     self._target_subject_id = '12'
        # elif 52 <= epoch < 54:
        #     self._target_subject_id = '13'
        # elif 54 <= epoch < 60:
        #     self._target_subject_id = '14'
        # elif 60 <= epoch < 62:
        #     self._target_subject_id = '15'
        # elif 62 <= epoch < 64:
        #     self._target_subject_id = '16'
        # elif epoch >= 64:

        print(" WE ARE TRAINING FOR SUBJECT ID = {}".format(self._target_subject_id))

        return super().on_batch_begin(batch, logs)

    def on_epoch_begin(self, epoch, logs={}):
        print('')

    def get_observations_per_epoch(self):
        return self._extract_data_size

    def get_features_count(self):
        # removing timestamp columns
        return len(self.cols_to_train) - 1

    def get_total_categories(self):
        return len(self.label_map.keys())


if __name__ == '__main__':
    generator = MobiFallGenerator('./dataset/*/*/*/*/*.txt', istrain=False)
    x, y = generator.get_batch(10 * 10)
    print(x.shape)

    print(y.shape)

    print(generator.get_observations_per_epoch())
    print(generator.get_features_count())
    print(generator.get_total_categories())
