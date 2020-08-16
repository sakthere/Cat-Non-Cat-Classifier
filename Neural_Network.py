import numpy as np
import matplotlib as plt
import pandas as pd
import h5py
import scipy
from IPython import get_ipython
from cnn_utils import load_dataset
from scipy import ndimage
from matplotlib import pyplot

def train_set():
    train_data = h5py.File("C:\\Users\\User\\PycharmProjects\\pyth\\scratch\\datasets\\train_catvnoncat.h5", "r")
    train_set_x_orig = np.array(train_data["train_set_x"][:])
    train_set_y_orig= np.array(train_data["train_set_y"][:])

    test_data = h5py.File("C:\\Users\\User\\PycharmProjects\\pyth\\scratch\\datasets\\test_catvnoncat.h5", "r")
    test_set_x_orig = np.array(test_data["test_set_x"][:])
    test_set_y_orig= np.array(test_data["test_set_y"][:])

    classes = np.array(test_data["classes"][:])

    train_set_y_origin = train_set_y_orig.reshape((1, train_set_x_orig.shape[0]))
    test_set_y_origin = test_set_y_orig.reshape((1, test_set_x_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

get_ipython().magic("Matplotlib inline")

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

index = 45
plt.pyplot.imshow(train_set_x_orig[index])
print("y = " + str(train_set_y[:, index])+ "it's a '"+ classes[np.squeeze(train_set_y[:,index])].decode("utf-8")+ " 'picture")

m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]

train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape, -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape, -1).T

train_set_x = train_set_x_flatten/255
test_set_x = test_set_x_flatten/255

def sigmoid(z):

    s = 1 / (1 + np.exp(-z))

    return s

def initialize_with_zeroes(dim):
    w = np.array([dim, 1])
    b = 0

    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))

    return w, b

dim = 3

w, b = initialize_with_zeroes(dim)
print("w =" + str(w))
print("b =" +str(b))


def propagate(w, b, X, Y):
    m = X.shape[0]

    A = sigmoid(np.dot(w.T, X)+b)

    cost= -1/m*np.sum(Y*np.log(A) + (1 - Y)*np.log(1 - A))

    dw = 1/m * np.dot(X, (A - Y).T)
    db = 1/m * np.sum(A, Y)

    assert(dw.shape == w.shape)
    assert(db.shape == b.shape)
    cost = np.squeeze(cost)
    assert(cost.shape == ())

    grads = {"dw" : dw,
             "db" : db}

    return grads, cost


def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):

    costs = []

    for i in range(num_iterations):

        grads, cost = propagate(w, b, X, Y)

        dw = grads["dw"]
        db = grads["db"]

        w = w - learning_rate*dw
        b = b- learning_rate*db

        if i%100 == 0:
            costs.append(cost)

        if print_cost and i%100==0:
            print("cost after {} iterations is {}: " .format(i, cost))

        params = { 'w' : w,
                   'b' : b
                   }
        grads = { 'dw': dw,
                  'db': db}

        return params, grads, costs

