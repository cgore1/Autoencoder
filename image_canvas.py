from PIL import Image
import os

size = 32 * 10
new_im = Image.new('L', (size,size))
# dir = 'TestImages/TestImages/'
dir = 'Weights/'
i = 0
j = 0
images = []
index = 100
for im in  (os.listdir(dir)):
    images.append(Image.open(dir + im))
    index = index - 1
    if(index == 0):
        break
index = 0
for i in range(0,10):
    for j in range(0, 10):
        new_im.paste(images[index], (i*32, j*32))
        index += 1
new_im.show()
new_im.save('output_test.bmp')