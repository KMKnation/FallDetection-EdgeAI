import getpass
import time
import numpy as np
from matplotlib import pyplot as plt
from send_mail import SendMail
from model import MobiFallNet
from DataGenerator import MobiFallGenerator
import os

sender = input("Type email address from which you want to send alerts: ")
password = getpass.getpass('Password:')
receiver = input("Type email address of whom you want to send alerts: ")

mailer = SendMail(sender, password, receivers=[receiver])
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

labels = {
    'STD': 'Standing',
    'WAL': 'Walking',
    'JOG': 'Jogging',
    'JUM': 'Jumping',
    'STU': 'Stairs up',
    'STN': 'Stairs down',
    'SCH': 'Sit chair',
    'CSI': 'Car-step in',
    'CSO': 'Car-step out',
    'FOL': 'Forward-lying',
    'FKL': 'Front-knees-lying',
    'BSC': 'Back-sitting-chair',
    'SDL': 'Sideward-lying'
}

label_map = list(label_map.keys())
# ['STD', 'WAL', 'JOG', 'JUM', 'STU', 'STN', 'SCH', 'CSI', 'CSO', 'FOL', 'FKL', 'BSC', 'SDL']


SENSOR_TO_TRAIN = ['acc', 'ori', 'gyro']

n_timestamps = 150

run_step = 2

x = [_ for _ in range(n_timestamps)]
ax = [0 for _ in range(n_timestamps)]
ay = [0 for _ in range(n_timestamps)]
az = [0 for _ in range(n_timestamps)]

# train the network
steps_per_epoch = 20
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


lastfall = None


def getFallDescription(type):
    if type == 'FOL':
        return 'Fall Forward from standing, use of hands to dampen fall'
    elif type == 'FKL':
        return 'Fall forward from standing, first impact on knees '
    elif type == 'BSC':
        return 'Fall backward while trying to sit on a chair'
    elif type == 'SDL':
        return 'Fall sidewards from standing, bending legs'


def draw_flow(cols, test_data, label, lastfall=None):
    start_time = time.time()

    plt.axis([0, 151, -20, 20])
    plt.ion()

    plt.show()

    steps = int(n_timestamps / run_step)

    # prediction = model.predict(np.array([[test_data]]))  # check input shape {batch size, timestamps, features}
    prediction = model.predict(
        np.array([test_data]).astype(np.float32))  # check input shape {batch size, timestamps, features}

    prediction = label_map[np.argmax(prediction)]
    if (prediction in ['BSC', 'FKL', 'FOL', 'SDL']):
        if lastfall != prediction:
            lastfall = prediction
            body = 'Description : {}'.format(getFallDescription(lastfall))
            fall = labels[prediction]
            mailer.send_alert(body, fall)
            print('Alert sent {}'.format(body))
    else:
        lastfall = None

    for i in range(steps):

        if i < steps:
            # predict = run.run(test_data[i * run_step - timestamps: i * run_step, :])
            title = 'correct: {}    predict: {}'.format(labels[label[i]], labels[prediction])

            # update_show_data(ax, run_step, test_data[i * run_step:i * run_step + run_step, 0])
            # update_show_data(ay, run_step, test_data[i * run_step:i * run_step + run_step, 1])
            # update_show_data(az, run_step, test_data[i * run_step:i * run_step + run_step, 2])
            update_show_data(ax, run_step, test_data[i * run_step:i * run_step + run_step, 0])
            update_show_data(ay, run_step, test_data[i * run_step:i * run_step + run_step, 1])
            update_show_data(az, run_step, test_data[i * run_step:i * run_step + run_step, 2])

            plt.cla()
            plt.plot(x, ax, label='{}'.format(cols[0]))
            plt.plot(x, ay, label='{}'.format(cols[1]))
            plt.plot(x, az, label='{}'.format(cols[2]))
            plt.legend()

            plt.title(title)
            plt.draw()
            plt.pause(0.001)
            inference_time = str(time.time() - start_time)
            print('Inference Time', inference_time)

    return lastfall


x_features, y_labels, cols = generator.get_test_data(subject_id=2)

for i in range(len(x_features)):
    lastfall = draw_flow(cols, x_features[i], y_labels[i], lastfall)
