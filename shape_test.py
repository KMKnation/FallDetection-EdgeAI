import numpy as np
batch_size = 10
timesteps = 1
datadim = 4
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