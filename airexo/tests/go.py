import numpy as np

path = '/home/ryan/Documents/GitHub/AirExo-2-test/data/calib_robot/1234567890.npy'
data = np.load(path, allow_pickle=True)
print(data)