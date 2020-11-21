from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import LSTM, Dropout, Dense, Flatten
from tensorflow.python.keras.optimizers import SGD


class MobiFallNet(object):

    def __init__(self, input_shape, n_outputs, pretrained_path=None):
        self.model = Sequential()
        self.model.add(LSTM(100, input_shape=input_shape))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(100, activation='relu'))
        self.model.add(Dense(n_outputs, activation='softmax'))
        optimizer = SGD(lr=0.01, nesterov=True)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    def get_model(self):
        print(self.model.summary())
        return self.model
