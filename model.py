from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import Conv1D, TimeDistributed, MaxPooling1D, Flatten


class MobiFallNet(object):

    def __init__(self, input_shape, n_outputs, pretrained_path=None):
        self.model = Sequential()
        self.model.add(LSTM(100, input_shape=input_shape, return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(100))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(100, activation='relu'))
        self.model.add(Dense(n_outputs, activation='softmax'))

        if pretrained_path != None:
            self.model.load_weights(pretrained_path)
            print('Loading weights from {}'.format(pretrained_path))
        optimizer = SGD(lr=0.001, momentum=0.9, nesterov=True)
        # adam = Adam(lr=0.001)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    def get_model(self):
        print(self.model.summary())
        return self.model


class CNNMobiFallNet(object):
    def __init__(self, n_timestamps, n_features, n_outputs, pretrained_path=None):
        self.model = Sequential()
        self.model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu'),
                                       input_shape=(None, n_timestamps, n_features)))
        self.model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu')))
        self.model.add(TimeDistributed(Dropout(0.5)))
        self.model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
        self.model.add(TimeDistributed(Flatten()))
        self.model.add(LSTM(100))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(100, activation='relu'))
        self.model.add(Dense(n_outputs, activation='softmax'))

        if pretrained_path != None:
            self.model.load_weights(pretrained_path)
            print('Loading weights from {}'.format(pretrained_path))
        # optimizer = SGD(lr=0.01, momentum=0.9, nesterov=True)
        adam = Adam(lr=0.001)
        self.model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    def get_model(self):
        print(self.model.summary())
        return self.model
