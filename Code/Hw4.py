## Stochastic gradient descent


import csv
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
from gurobipy import *
import numpy as np
from numpy import *


def csv_reader(file_obj):
    reader = csv.reader(file_obj)
    data = []
    for row in reader:
        data.append(" ".join(row))
    return data


def sanitize(path):
    data_set = []
    with open(path, "rb") as f_obj:
        data = csv_reader(f_obj)

    for item in data:
        set = []
        for num in item.split(" "):
            set.append(float(num))
        data_set.append([set[0], set[1], set[2]])
    return data_set

def sigmoid(score):
    return (1 / (1 + np.exp(-1*score)))

def cost(theta, X, y):
    sxt = sigmoid(y*np.dot(X, theta));
    mcost = 1 + np.exp(sxt)
    return log(mcost.mean())

def gradient(theta, X, y):
    sxt = sigmoid(np.dot(X, theta))

    err = sxt - y
    grad = np.dot(err, sxt) / y.size
    return grad


def log_reg(learning_rate, Y, X, gamma, conv):
    theta = theta = 0.1* np.random.randn(2);
    while(True):
        print "theta is " + str(theta)

        d = gradient(theta, X, Y)
        print "delta is " + str(d)
        old_cost = cost(theta, X, Y)
        theta = np.subtract(theta, learning_rate*d)
        cur_cost = cost(theta, X, Y)
        print "Abs Cost diff is " + str(abs(old_cost - cur_cost))
        if (abs(old_cost - cur_cost) < conv):
            break
        # print "obj is " + str(obj(theta, X, Y))

    return theta


def sign(theta_X, theta_Y, item):
    if theta_X * item[0] + theta_Y * item[1] > 0:
        return 1
    else: return -1

def vectorise(path):
    pos_arr = []
    neg_arr = []
    for item in sanitize(path):
        if item[2] > 0:
            pos_arr.append([item[0], item[1]])
        else:
            neg_arr.append([item[0], item[1]])

    data_dict = {1:np.asarray(pos_arr), -1:np.asarray(neg_arr)}
    return data_dict

def predict(test_data_set, theta_X, theta_Y):
    hits = 0
    for item in test_data_set:
        if (sign(theta_X, theta_Y, item)) == item[2]:
            hits = hits + 1

    print "Hits :- " + str(hits)
    print "Misses :- " + str(len(test_data_set) - hits)
    print "Total :- " + str(len(test_data_set))
    print "Accuracy is " + str(float(hits)/float(len(test_data_set)) * 100) + "%"


if __name__ == '__main__':
    DATASET = "B"
    gamma = 1
    learning_rate = 0.001
    conv = 0.0001

    train_path = os.getcwd() + "/data/" + DATASET + "/train.csv"
    test_path = os.getcwd() + "/data/" + DATASET + "/test.csv"
    train_data_dict = vectorise(train_path)
    train_data_set = sanitize(train_path)
    test_data_set = sanitize(test_path)
    test_data_dict = vectorise(test_path)
    X_train = []
    Y_train = []
    for data in train_data_set:
        X_train.append([data[0], data[1]])
        Y_train.append(data[2])

    theta = log_reg(learning_rate, np.asarray(Y_train), np.asarray(X_train), gamma, conv)
    print "Final theta is " + str(theta)
    predict(test_data_set, theta[0], theta[1])

    # plot_dict(train_data_dict, 8.2074795747618232, -4.926031566883299, gamma, DATASET)