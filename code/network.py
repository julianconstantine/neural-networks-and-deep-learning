"""
network.py
----------

A module to implement the stochastic gradient descent learning algorithm for a feedforward neural network. Gradients
are calculated using backpropagation.
"""

# Import libraries
import numpy as np

import random


class Network(object):
    def __init__(self, sizes):
        """The list 'sizes' contains the number of neurons in the respective layers of the network. For example,
        if the last was [2, 3, 1] then it would be a three-layer network, with the first layer containing 2 neurons,
        the second layer 3 neurons, and the third layer 1 neuron. The biases and weights for the network are
        initialized randomly, using a Gaussian distribution with mean 0 and variance 1. Note that the first layer is
        assumed to be an input layer, and by convention we won't set any biases for those neurons, and biases are
        only ever used in computing the outputs from layer layers."""

        # Number of layers
        self.num_layers = len(sizes)

        # Number of neurons in each later
        self.sizes = sizes

        # Initialize biases for each of the non-input layers to random normally distributed values
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]

        # Initialize weights using random normally distributed values
        # self.weights[i][j][k] is the weight on the connection between the kth neuron in the (i+1)st later and the
        # jth neuron in the (i+2)nd layer (where the layers are indexed starting from 0)
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if a is the input"""
        for b, w in zip(self.biases, self.weights):
            # Feedthe input forward through each layer of the network
            a = sigmoid(np.dot(w, a) + b)

        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """Train the neural network using mini-batch stochastic gradient descent. The 'training data' is a list of
        tuples (x, y) representing the training inputs and the desired outputs. The other non-optional parameters are
        self-explanatory. If 'test_data' is provided then the network will evaluated against the test data at each
        epoch, and partial progress printed out. This is useful for tracking progress, but slows things down
        dramatically"""
        if test_data: n_test = len(test_data)

        n = len(training_data)

        for j in range(epochs):
            random.shuffle(training_data)

            mini_batches = [training_data[k:(k+mini_batch_size)] for k in range(0, n, mini_batch_size)]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            if test_data:
                print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying gradient descent using backpropagation to a single
        mini batch. The 'mini_batch' is a list of tuples (x, y) and 'eta' is the learning rate"""
        # Initialize gradients w/rt b and w as all zeroes
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)

            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [w - (eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


net = Network([2, 3, 1])
