import numpy as np
import data_load as dl
from PIL import Image
import glob, os


numLayers = 3
# layerNodes = [1024, 512, 1024]
layerNodes = [1024,4,1024]
expected_output = [[0.1, 0.2, 0.3, 0.4]]
learning_rate = 0.01

def sigmoid(z):
    z = 1.0/ (1 + np.exp(-z))
    return z

bias = []
for y in layerNodes[1:]:
    bias.append(np.random.randn(y,1))
    # print np.random.randn(y,1)

weight = []
for x,y in zip(layerNodes[1:], layerNodes[:-1]):
    weight.append(np.random.randn(x, y))

def forward_propogation(input):
    output = input
    forward_list = []
    zlist = []
    for b, w in zip(bias, weight):
        z = np.dot(output, np.transpose(w)) + np.transpose(b)
        zlist.append(z)
        output = sigmoid(z)
        forward_list.append(np.array(output))

    return forward_list

def backpropagate(input, y):
    # forward
    output = input
    forward_list = [input]
    zlist = []
    for b, w in zip(bias, weight):
        z = np.dot(output, np.transpose(w)) + np.transpose(b)
        zlist.append(z)
        output = sigmoid(z)
        forward_list.append(np.array(output))

    # back
    # final layer
    # print len(forward_list)
    delta = (y - forward_list[-1]) * (sigmoid(zlist[-1])* sigmoid(1 - zlist[-1]))
    deltalist = [delta]
    prevDelta = delta
    partialDvsW = [np.zeros(w.shape) for w in weight]
    partialDvsB = [np.zeros(b.shape) for b in bias]
    partialDvsW[-1] = np.dot(np.transpose(delta), forward_list[-2])
    partialDvsB[-1] = np.transpose(delta)

    for i in range(2, len(forward_list)):
        # print 'wt - ' + str(weight[-i])
        # print 'fw - ' + str(forward_list[-i+1])
        # print (i)
        prod = np.dot( prevDelta, weight[-i + 1])
        # print 'prev -'
        # print prevDelta
        # print 'wt -'
        # print weight[-i + 1]

        delta = prod * (sigmoid(zlist[-i]) * (1 - sigmoid(zlist[-i])))
        prevDelta = delta
        partialW = np.dot(np.transpose(delta), forward_list[-i -1])
        # print 'partialW - '
        # print partialW
        partialDvsW[-i] = partialW
        partialDvsB[-i] = np.transpose(delta)

        deltalist.insert(0, delta)
        # d = np.dot(np.transpose(forward_list[-i]), delta)
        # print 'd = ' + str(d)

    # for w in weight:
    #     print 'w - ' + str(w)
    # for f in forward_list:
    #     print 'f - ' + str(f)
    # for l in deltalist:
    #     print 'delta - ' + str(l)
    # for l in partialDvsW:
    #     print 'partialW - ' + str(l)
    # for l in partialDvsB:
    #     print 'partialB - ' + str(l)
    # print 'w '
    # print weight
    # print 'partial W'
    # print partialDvsW
    # print 'diff '
    # print np.array(weight) - np.array(partialDvsW)
    # print 'diff b'
    # print np.array(bias)
    # print partialDvsB
    # print np.array(partialDvsB)

    # for w,partialW in zip(weight, partialDvsW):
    #     print w
    #     print partialW
    #     print '--'

    # print weight - partialDvsW
    return partialDvsW, partialDvsB, output

# input = np.array([[0.4,0.3,0.2,0.1]])
new_input = dl.get_img()
input = [new_input]
expected_output = input

prev_output=[[]]
error_val=[[1.0,1.0,1.0,1.0]]

for i in range(0, 10000):
    partialW, partialB,output = backpropagate(input, expected_output)
    weight = np.array(weight) + learning_rate * np.array(partialW)
    bias = np.array(bias) + learning_rate * np.array(partialB)
    # if prev_output==output:
    #     break
    # else:
    #     prev_output=output
    # error_val[0][0] = (expected_output[0][0]-output[0][0])/input[0][0]
    # error_val[0][1] = (expected_output[0][1] - output[0][1])/input[0][0]
    # error_val[0][2] = (expected_output[0][2] - output[0][2])/input[0][0]
    # error_val[0][3] = (expected_output[0][3] - output[0][3])/input[0][0]

    # print "error is:",(error_val[0])

    # print (expected_output[0][0])
    # if(error_val[0][0]<=0.001 and error_val[0][1]<=0.001 and error_val[0][2]<=0.001 and error_val[0][3]<=0.001):
    #     break
    if i % 100 is 0:
        print output - input

    size = 32, 32

result = Image.fromarray(output*256)
result.show()
result.save('out.bmp')

# for infile in glob.glob("*.pgm"):
#     file, ext = os.path.splitext(infile)
#     im = Image.open(infile)
#     im.thumbnail(size, Image.ANTIALIAS)
#     im.save(file + ".thumbnail", "pgm")



res = []
index = 0
for i in range(0,32):
    my_row = []
    for j in range(0,32):
        my_row.append(output[0][index] * 256)
        index += 1
    res.append(my_row)



#
# print weight



