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

def get_img():
    return images


