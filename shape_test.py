import numpy as np

from model import MobiFallNet

batch_size = 10
timesteps = 1
datadim = 3
nb_classes = 13
x_train = np.random.random((batch_size, timesteps, datadim))

print(x_train.shape)

y_train = np.random.random((batch_size, nb_classes))

print(y_train.shape)


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


print(list(label_map)[5])
n_features = 3
n_category = 13
n_timestamps = 30
input_shape = (n_timestamps, n_features)
model = MobiFallNet(input_shape=input_shape, n_outputs=n_category).get_model()
print(model.input)

prediction = model.predict(np.array([x_train[0]]))
print(prediction)
print(np.argmax(prediction))