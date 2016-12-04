import numpy as np
import data_load as dl
from PIL import Image
import glob, os
from scipy.misc import toimage
import math

bias_term = 1
numLayers = 3
# layerNodes = [1024, 512, 1024]
pixels = 1024
layerNodes = [pixels + 1 , 512 + 1, pixels]
learning_rate = 0.001
images_in_batch = 50
file_bias = 'single_bias.npy'
file_weights = 'single_weights.npy'

def sigmoid(z, shouldModifyLast= False):
    z = 1.0/ (1 + np.exp(-z))
    if shouldModifyLast:
        z[0][-1] = bias_term
    return z

    weight = np.load('weights.npy')


bias = []
if os.path.exists(file_bias):
    bias = np.load(file_bias)
else:
    for y in layerNodes[1:]:
        bias.append(np.random.randn(y,1))

weight = []
if os.path.exists(file_weights):
    weight = np.load(file_weights)
else:
    for x,y in zip(layerNodes[1:], layerNodes[:-1]):
        weight.append(np.random.randn(x, y))

np.set_printoptions(threshold=np.inf)

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

def forward_pass_tester():
    for img in dl.get_test_img():
        my_input = []
        for r in img:
            my_input.append(r)
        input = [my_input]
        input[0].append(float(bias_term))
        expected_output = my_input[:len(my_input) - 1]
        out = forward_propogation(input)
        err = out[-1][0] - expected_output
        # print err
        print  np.dot(err, np.transpose(err)) / len(err)

# forward_pass_tester()

def backpropagate(input, y):
    # forward
    output = input
    forward_list = [input]
    zlist = []
    layer = 0
    for b, w in zip(bias, weight):
        z = np.dot(output, np.transpose(w)) + np.transpose(b)
        zlist.append(z)
        if layer is len(layerNodes) - 1:
            output = sigmoid(z, False)
        else:
            output = sigmoid(z, True)
        forward_list.append(np.array(output))
        layer += 1

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

    return partialDvsW, partialDvsB, output

def visualize(output):
    res = []
    index = 0

    for i in range(0, 32):
        my_row = []
        for j in range(0, 32):
            my_row.append(int(output[0][index] * 255))
            index += 1
        res.append(my_row)
    res = np.transpose(np.array(res))
    toimage(res).show()

def batch_update():
    global weight
    global bias
    start = 0

    for start in range(0, len(dl.get_img()), images_in_batch):
        print '--- Strting batch ' + str(start / images_in_batch + 1), start
        sumerr = 1000
        error_threshold = 0.0001
        prev_sumerr = sumerr + images_in_batch
        while (prev_sumerr - sumerr) / images_in_batch > error_threshold:
            prev_sumerr = sumerr
            sumerr = 0

            for i in range(start, start + images_in_batch):
                new_input = []
                for r in dl.get_img()[i]:
                    new_input.append(r)
                input = [new_input]
                input[0].append(float(bias_term))
                expected_output = [new_input[:len(new_input) - 1]]
                prev_output=[[]]

                partialW, partialB,output = backpropagate(input, expected_output)
                wt = np.array(weight) + learning_rate * np.array(partialW)
                weight = wt
                bs = np.array(bias) + learning_rate * np.array(partialB)
                bias=bs
                error = np.array((output - [input[0][:len(new_input) - 1]])[0])
                sumerr = sumerr + np.dot(error, np.transpose(error)) / len(error)

                # if i % 100 is 0:
                #     a = np.array((output - [input[0][:len(new_input) - 1]])[0])
                #     # print a
                #     print np.dot(a, np.transpose(a)) / len(a)
            print sumerr / images_in_batch

        print '--- batch ' + str(start / images_in_batch + 1) + ' done!'

def saveWeights():
    np.save(file_weights, np.array(weight))
    np.save(file_bias, np.array(bias))

def train():
    for i in range(0, 1000):
        print '***'
        print i
        print '***'
        batch_update()
        saveWeights()

train()



# new_input = []
# for r in dl.get_img()[0]:
#     new_input.append(r)
# input = [new_input]
# input[0].append(float(bias_term))
# expected_output = [new_input[:len(new_input) - 1]]
# prev_output = [[]]
# partialW, partialB, output = backpropagate(input, expected_output)
#
# visualize(output)
#
# with open('weights', 'w') as f:
#     for w in weight:
#         f.write(str(w))
#
# with open('bias', 'w') as f:
#     for b in bias:
#         f.write(str(b))

def visualizeWeights():
    ar = np.zeros(pixels)

    for i in range(0 ,1):
        w = weight[0][i]
        den = math.sqrt(np.dot(w, np.transpose(w)))
        for j in range(0, pixels):
            ar[j] = w[j] / den
        visualize([np.asarray(ar)])
# visualizeWeights()
