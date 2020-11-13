from tensorflow.python.keras.callbacks import ModelCheckpoint

from model import MobiFallNet
from DataGenerator import MobiFallGenerator

generator = MobiFallGenerator('./dataset/*/*/*/*/*.txt',
                              train_for='acc',
                              extract_data_size=30,
                              istrain=True,
                              ratio=0.3)

validation_generator = MobiFallGenerator('./dataset/*/*/*/*/*.txt',
                                         train_for='acc',
                                         extract_data_size=30,
                                         istrain=False,
                                         ratio=0.3)

n_observations = generator.get_observations_per_epoch()
n_features = generator.get_features_count()
n_category = generator.get_total_categories()

input_shape = (n_observations, n_features)
model = MobiFallNet(input_shape=input_shape, n_outputs=n_category).get_model()

# train the network
steps_per_epoch = 10

callbacks_list = [ModelCheckpoint('weights.hdf5', monitor='loss', verbose=1, save_best_only=True, mode='auto',
                                  save_weights_only='True')]
H = model.fit_generator(generator.get_batch(30),
                        steps_per_epoch=steps_per_epoch,
                        epochs=1,
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
