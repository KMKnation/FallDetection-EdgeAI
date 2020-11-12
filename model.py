from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import LSTM, Dropout, Dense


class MobiFallNet(object):

    def __init__(self, n_timesteps, n_features, n_outputs, pretrained_path=None):
        self.model = Sequential()
        self.model.add(LSTM(100, input_shape=(n_timesteps, n_features)))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(100, activation='relu'))
        self.model.add(Dense(n_outputs, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def get_model(self):
        return self.model
