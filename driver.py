import time
import numpy as np
from matplotlib import pyplot as plt

timestamps = 30
run_step = 2

batch_size = 30
datadim = 4
nb_classes = 13
x_train = np.random.random((batch_size, timestamps, datadim))


def update_show_data(data, step, update_data):
    for i in range(step):
        data.pop(0)
        data.append(update_data[i])


def draw_flow(test_data, test_label):
    data_size = test_data.shape[0]
    x = [_ for _ in range(timestamps)]
    ax = [0 for _ in range(timestamps)]
    ay = [0 for _ in range(timestamps)]
    az = [0 for _ in range(timestamps)]

    start_time = time.time()

    plt.axis([0, 151, -20, 20])
    plt.ion()

    plt.show()

    steps = int(timestamps / run_step)

    for i in range(steps):

        if i < steps:
            # predict = run.run(test_data[i * run_step - timestamps: i * run_step, :])
            title = 'correct:' + 'Correct' + '     predict:' + 'Predicted'

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
            during = str(time.time() - start_time)
            print('Inference Time', during)


draw_flow(x_train[0], '')
