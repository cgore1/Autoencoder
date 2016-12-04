import numpy as np
from PIL import Image
import numpy
import os

dir = 'TrainImages/TrainImages/'
images = []
def load_img():
    for im in  (os.listdir(dir)):
        img  = Image.open(dir + im)
        pixel = img.load()
        my_arr = []
        w = img.size[0]
        h = img.size[1]
        for i in range(w):
            for j in range(h):
                my_arr.append((pixel[i, j] / 256.0))
        images.append(my_arr)

load_img()


dir_test = 'TestImages/TestImages/'
test_images=[]
def load_test_img():
    for im in  (os.listdir(dir)):
        img  = Image.open(dir + im)
        pixel = img.load()
        my_new_arr=[]
        w1 = img.size[0]
        h1 = img.size[1]
        for i in range(w1):
            for j in range(h1):
                my_new_arr.append((pixel[i, j] / 256.0))
        test_images.append(my_new_arr)

load_test_img()


def get_test_img():
    return test_images

def get_img():
    return images


