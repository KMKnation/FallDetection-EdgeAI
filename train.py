from tensorflow.python.keras.callbacks import ModelCheckpoint
from model import MobiFallNet
from DataGenerator import MobiFallGenerator
import matplotlib.pyplot as plt
import os

ROOT_DIRECTORY = os.getcwd()

n_timestamps = 10

# train the network
steps_per_epoch = 20
epochs = 50
batchsize = steps_per_epoch * epochs

generator = MobiFallGenerator('./dataset/*/*/*/*/*.txt',
                              train_for='acc',
                              batch_size=batchsize,
                              extract_data_size=n_timestamps,
                              istrain=True,
                              ratio=0.3)

validation_generator = MobiFallGenerator('./dataset/*/*/*/*/*.txt',
                                         train_for='acc',
                                         batch_size=batchsize,
                                         extract_data_size=n_timestamps,
                                         istrain=False,
                                         ratio=0.3)

n_features = generator.get_features_count()
n_category = generator.get_total_categories()

input_shape = (n_timestamps, n_features)
print("INPUT SHAPE =>{}".format(str(input_shape)))
model = MobiFallNet(input_shape=input_shape, n_outputs=n_category).get_model()

print(model.summary())

callbacks_list = [
    ModelCheckpoint(os.path.join('model', 'weights.hdf5'), monitor='loss', verbose=1, save_best_only=True, mode='auto',
                    save_weights_only='True')]

history = model.fit_generator(generator.next_train(),
                              steps_per_epoch=steps_per_epoch,
                              epochs=epochs,
                              verbose=1,
                              callbacks=callbacks_list,
                              validation_data=validation_generator.next_val(),
                              validation_steps=int((steps_per_epoch // 4)),
                              validation_freq=1,
                              max_queue_size=10,
                              workers=1,
                              use_multiprocessing=True,
                              shuffle=False,
                              initial_epoch=0)

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.savefig(os.path.join(os.path.join(ROOT_DIRECTORY, 'train_logs'), 'accuracy.png'))
plt.clf()

plt.figure()

plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.savefig(os.path.join(os.path.join(ROOT_DIRECTORY, 'train_logs'), 'loss.png'))

plt.clf()

exit(0)