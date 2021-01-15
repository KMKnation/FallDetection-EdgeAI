from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.callbacks import LearningRateScheduler

from model import MobiFallNet, CNNMobiFallNet
from DataGenerator import MobiFallGenerator, MobiDataGenerator
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import KFold
import numpy as np
from tensorflow.python.keras.utils.np_utils import to_categorical
import tensorflow as tf

ROOT_DIRECTORY = os.getcwd()

SENSOR_TO_TRAIN = ['acc', 'ori', 'gyro']

n_timestamps = 800

# train the network
epochs = 1000
# batchsize = steps_per_epoch * epochs
batchsize = 26

lr = 0.001


def lr_schedule(epoch):
    return lr * (0.1 ** int(epoch / 10))


generator = MobiDataGenerator('./dataset/*/*/*/*/*.txt',
                              train_for=SENSOR_TO_TRAIN[0],
                              batch_size=batchsize,
                              extract_data_size=n_timestamps,
                              istrain=True,
                              ratio=0.2)

X, y = generator.get_data()

n_category = generator.get_total_categories()

kf = KFold(n_splits=5)
fold_count = 0
for train_index, val_index in kf.split(X, y):
    fold_count += 1

    trainX = np.array([X[i] for i in train_index])
    trainY = to_categorical(y.take(train_index), num_classes=generator.get_total_categories())

    testX = np.array([X[i] for i in val_index])
    testY = to_categorical(y.take(val_index), num_classes=generator.get_total_categories())

    input_shape = (X.shape[1], X.shape[2])

    model = MobiFallNet(input_shape=input_shape, n_outputs=n_category, lr=lr).get_model()

    print(model.summary())

    callbacks_list = [
        ModelCheckpoint(os.path.join('model', 'weights_{}.hdf5'.format(fold_count)), monitor='loss', verbose=1,
                        save_best_only=True, mode='auto',
                        save_weights_only='True')]

    steps_per_epoch = len(trainY) / batchsize

    history = model.fit(x=trainX,
                        y=trainY,
                        epochs=epochs,
                        verbose=2,
                        steps_per_epoch=steps_per_epoch,
                        callbacks=callbacks_list,
                        validation_data=(testX, testY),
                        shuffle=True,
                        validation_freq=2,
                        workers=2,
                        use_multiprocessing=True)

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.plot(history.epoch, acc, 'b', label='Training acc')
    plt.plot(history.epoch, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.savefig(os.path.join(os.path.join(ROOT_DIRECTORY, 'train_logs'), 'accuracy_{}.png'.format(fold_count)))
    plt.clf()

    plt.figure()

    plt.plot(history.epoch, loss, 'b', label='Training loss')
    plt.plot(history.epoch, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.savefig(os.path.join(os.path.join(ROOT_DIRECTORY, 'train_logs'), 'loss{}.png'.format(fold_count)))

    plt.clf()
