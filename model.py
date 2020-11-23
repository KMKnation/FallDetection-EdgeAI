from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import SGD


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
        optimizer = SGD(lr=0.01, nesterov=True)
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def get_model(self):
        print(self.model.summary())
        return self.model
