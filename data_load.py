import numpy as np
from PIL import Image
import numpy


# img = Image.open("/Users/Roshni/Downloads/TrainImages/Ashanti_0003.pgm")



def get_img():
    my_arr = []
    my_file = '/Users/Roshni/Downloads/TrainImages/Ashanti_0003.pgm'  # image can be in gif jpeg or png format
    img = Image.open(my_file)
    # img.show()
    print "img", img
    pixel = img.load()
    w = img.size[0]
    h = img.size[1]
    for i in range(w):
        for j in range(h):
            my_arr.append((pixel[i, j] / 256.0))

    return my_arr


get_img()
