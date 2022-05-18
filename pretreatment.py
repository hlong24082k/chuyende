from array import array
import os
import re
from xml.etree.ElementTree import tostring
import cv2 
import numpy as np

# path = "archive/flowers/"
path = "data/"
# path_image = "archive/flowers/daisy/5547758_eea9edfd54_n.jpg"
        
def convert_gray(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

def resize_image(image):
    image = cv2.resize(image,(64,64))
    return image

def convert_reshape_array(image):
    image_array = np.array(image)
    image_array = np.reshape(image_array,(1,-1))
    return image_array

def list_path_folders(path):
    list_path_folder = []
    for folder in os.listdir(path):
        current_folder = os.path.join(path, folder)
        list_path_folder.append(current_folder)
    return list_path_folder    

def list_images(path):
    list_image = []
    for image in os.listdir(path):
        current_image = os.path.join(path,image)
        list_image.append(current_image)
    return list_image

def list_array_images (list_images):
    list = np.arange(4096).reshape(1,4096)
    for im in list_images:
        im = cv2.imread(im)
        im = convert_gray(im)
        im = resize_image(im)
        im = convert_reshape_array(im)
        list = np.concatenate((list, im), axis=0)
        lists = np.delete(list, 0, 0)
    return lists

def list_imread_path(list_path_folders):
    for folder in list_path_folders:
        list_image = list_images(folder)
        list_array_image = list_array_images(list_image)    
        np.savetxt("./data_pre/" + folder.split("/")[-1] + ".txt", list_array_image)   

if __name__ == "__main__":
    list_path_folder = list_path_folders(path)
    list_imread_path(list_path_folder)

# path_folder_rose = "archive/flowers/rose"
# list_images = list_images(path_folder_rose)
# list_array_images = list_array_images(list_images)    
# np.savetxt("./data/" + path_folder_rose.split("/")[-1] + ".txt", list_array_images)  
# list_imread_path(path_folder_rose)



# list_imread_path(list_folder)