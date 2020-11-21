from tensorflow.python.keras.callbacks import ModelCheckpoint

from model import MobiFallNet
from DataGenerator import MobiFallGenerator

n_timestamps = 5
generator = MobiFallGenerator('./dataset/*/*/*/*/*.txt',
                              train_for='acc',
                              extract_data_size=n_timestamps,
                              istrain=True,
                              ratio=0.3)

validation_generator = MobiFallGenerator('./dataset/*/*/*/*/*.txt',
                                         train_for='acc',
                                         extract_data_size=n_timestamps,
                                         istrain=False,
                                         ratio=0.3)

n_features = generator.get_features_count()
n_category = generator.get_total_categories()

input_shape = (n_timestamps, n_features)
print("INPUT SHAPE =>{}".format(str(input_shape)))
model = MobiFallNet(input_shape=input_shape, n_outputs=n_category).get_model()

print(model.summary())

# train the network
steps_per_epoch = 10
epochs = 100
batchsize = steps_per_epoch * epochs


callbacks_list = [ModelCheckpoint('weights.hdf5', monitor='loss', verbose=1, save_best_only=True, mode='auto',
                                  save_weights_only='True')]
H = model.fit_generator(generator.get_batch(batchsize),
                        steps_per_epoch=steps_per_epoch,
                        epochs=epochs,
                        verbose=1,
                        callbacks=callbacks_list,
                        validation_data=validation_generator.get_batch(10),
                        validation_steps=int((steps_per_epoch // 4)),
                        validation_freq=1,
                        max_queue_size=10,
                        workers=1,
                        use_multiprocessing=True,
                        shuffle=False,
                        initial_epoch=0)
