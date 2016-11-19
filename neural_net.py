import numpy as np

numLayers = 3
# layerNodes = [1024, 512, 1024]
layerNodes = [4,2,4]
expected_output = [[10.0, 10.0, 10.0, 10.0]]

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
    for b, w in zip(bias, weight):
        z = np.dot(output, np.transpose(w)) + np.transpose(b)
        output = sigmoid(z)
        forward_list.append(np.array(output))

    return forward_list
input = np.array([[10.0, 10.0, 10.0, 10.0]])
print forward_propogation(input)

