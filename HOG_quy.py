from importlib.resources import path
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import os

sell_size = np.array((8, 8))
block_size = np.array((16, 16))


def calculator_cell(gama, theta):
    gama = gama/20
    cell = np.zeros(((8, 8 ,9)))
    for r in range(0, 8):
        for c in range(0, 8):
            bin = np.zeros(9, dtype=float)
            for i in range(0, 8):
                for j in range(0, 8):
                    pos = theta[i + r * 8, j + c * 8]//20
                    bin[int(pos)] = np.float64(gama[i + r * 8, j + c * 8]) * abs(20 * (pos+1) - theta[i + r * 8, j + c * 8])
                    if pos+1 < 9:
                        bin[int(pos+1)] = gama[i + r * 8, j + c * 8] * abs(20 * pos-theta[i + r * 8, j + c * 8])
                    else:
                        bin[0] = gama[i + r * 8, j + c * 8] * abs(20 * pos-theta[i + r * 8, j + c * 8])
                cell[r, c, :] = bin
    return cell


def calculator_block(cell):
    block = np.zeros((( 7, 7, 36 )))
    for r in range(0, 7):
        for c in range(0, 7):
            bins = np.zeros(36)
            if np.sum(cell[r:r+2, c:c+2]) != 0:
                bins = np.reshape(cell[r:r+2, c:c+2], (1, 36)) / np.sum(cell[r:r+2, c:c+2])
            block[r, c, :] = bins
    return block

def calculator_mapping(X):
    Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    gx = cv2.filter2D(X, -1, Gx)
    gy = cv2.filter2D(X, -1, Gy)
    gama = np.sqrt(gx ** 2 + gy ** 2)
    theta = np.ones(gy.shape) * np.pi / 2
    theta[gx != 0] = np.arctan(gy[gx != 0] / gx[gx != 0])
    theta = theta * 180 / np.pi
    return gama, theta

def HOG(X):
    rows, columns = X.shape
    HOG = np.zeros((rows, 1764))
    for i in range(0, rows):
        img = np.reshape(X[i, :], (64, 64))
        gama, theta = calculator_mapping(img)
        cell = calculator_cell(gama, theta)
        block = calculator_block(cell)
        block = np.reshape(block, (1, 1764))
        HOG[i, :] = block
    return HOG

def list_path_folders(path):
    list_path_folder = []
    for folder in os.listdir(path):
        current_folder = os.path.join(path, folder)
        list_path_folder.append(current_folder)
    return list_path_folder  

if __name__ == "__main__":
    #doc du lieu data_pre
    path = "data_pre"
    list_path_data_pre = list_path_folders(path)

    #
    for txt in list_path_data_pre:
        print(txt)
        data_pre = np.loadtxt(txt)
        data_hog = HOG(data_pre)
        np.savetxt("./data_hog/" + txt.split("/")[-1], data_hog)

    # list_daisy = np.loadtxt("data_pre/daisy.txt")
    # list_daisy_hog = HOG(list_daisy)
    # x, y = list_daisy_hog.shape
    # print(x)
    # print(y)
    # np.savetxt("./data_hog/daisy.txt", list_daisy_hog)