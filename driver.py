import time
import numpy as np
from matplotlib import pyplot as plt
from model import MobiFallNet

from model import MobiFallNet
from DataGenerator import MobiFallGenerator
import os

ROOT_DIRECTORY = os.getcwd()

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

label_map = list(label_map.keys())
# ['STD', 'WAL', 'JOG', 'JUM', 'STU', 'STN', 'SCH', 'CSI', 'CSO', 'FOL', 'FKL', 'BSC', 'SDL']


SENSOR_TO_TRAIN = ['acc', 'ori', 'gyro']

n_timestamps = 30

run_step = 1

# train the network
steps_per_epoch = 10
epochs = 1
batchsize = steps_per_epoch * epochs

generator = MobiFallGenerator('./dataset/*/*/*/*/*.txt',
                              train_for=SENSOR_TO_TRAIN[0],
                              batch_size=batchsize,
                              extract_data_size=n_timestamps,
                              istrain=False,
                              ratio=0.3)

n_features = generator.get_features_count()
n_category = generator.get_total_categories()

input_shape = (n_timestamps, n_features)
print("INPUT SHAPE =>{}".format(str(input_shape)))

weights = os.path.join(os.path.join(ROOT_DIRECTORY, 'model'), 'weights.hdf5')
model = MobiFallNet(input_shape=input_shape, n_outputs=n_category, pretrained_path=weights).get_model()


# batch_size = 30
# datadim = 4
# nb_classes = 13
# x_train = np.random.random((batch_size, timestamps, datadim))


def update_show_data(data, step, update_data):
    for i in range(step):
        data.pop(0)
        data.append(update_data[i])


def draw_flow(test_data, label):
    x = [_ for _ in range(n_timestamps)]
    ax = [0 for _ in range(n_timestamps)]
    ay = [0 for _ in range(n_timestamps)]
    az = [0 for _ in range(n_timestamps)]

    start_time = time.time()

    plt.axis([0, 151, -20, 20])
    plt.ion()

    plt.show()

    steps = int(n_timestamps / run_step)

    for i in range(steps):

        if i < steps:
            # prediction = model.predict(np.array([[test_data]]))  # check input shape {batch size, timestamps, features}
            prediction = model.predict(np.array([test_data]))  # check input shape {batch size, timestamps, features}

            # predict = run.run(test_data[i * run_step - timestamps: i * run_step, :])
            title = 'correct: {}    predict: {}'.format(label_map[np.argmax(label)], label_map[np.argmax(prediction)])

            # update_show_data(ax, run_step, test_data[i * run_step:i * run_step + run_step, 0])
            # update_show_data(ay, run_step, test_data[i * run_step:i * run_step + run_step, 1])
            # update_show_data(az, run_step, test_data[i * run_step:i * run_step + run_step, 2])
            update_show_data(ax, run_step, test_data[i * run_step:i * run_step + run_step, 0])
            update_show_data(ay, run_step, test_data[i * run_step:i * run_step + run_step, 1])
            update_show_data(az, run_step, test_data[i * run_step:i * run_step + run_step, 2])

            plt.cla()
            plt.plot(x, ax)
            plt.plot(x, ay)
            plt.plot(x, az)

            plt.title(title)
            plt.draw()
            plt.pause(0.001)
            inference_time = str(time.time() - start_time)
            print('Inference Time', inference_time)


x, y = generator.get_batch(30, True)
for i in range(len(x)):
    draw_flow(x[i], y[i])
