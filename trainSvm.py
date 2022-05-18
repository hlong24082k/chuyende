from importlib.resources import path
import os
import numpy as np


def list_path_folders(path):
    list_path_folder = []
    for folder in os.listdir(path):
        current_folder = os.path.join(path, folder)
        list_path_folder.append(current_folder)
    return list_path_folder  


# path = "data_hog"
# list_path_data_hog = list_path_folders(path)
# print(list_path_data_hog)
# data = np.zeros((1,1764))
# for txt in list_path_data_hog:
#     data_hog = np.loadtxt(txt)
#     data = np.concatenate((data, data_hog), axis=0)
# np.savetxt("./data_hog/data_hog.txt", data)

###################################################
###################################################

#read data_hog
path = "data_hog/data_hog.txt"
data_hog = np.loadtxt(path)
print(data_hog[1, :])

#gan nhan
data_target = np.zeros((3500, 1))
print(data_target)