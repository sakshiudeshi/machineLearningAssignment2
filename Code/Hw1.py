#!/usr/bin/env python
import csv
import os
import matplotlib.pyplot as plt
from matplotlib import style
import pylab
import numpy as np
style.use('ggplot')
from gurobipy import *


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


def plot_dict(dict):
    colors = list("rb")

    for arr in dict.values():
        (x, y) = arr.T
        plt.scatter(x, y, color=colors.pop())

    plt.legend(dict.keys())
    plt.show()


def solve_primal(data_set, gamma):
    m = Model()

    theta = [m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="t0"),
             m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="t1")]


    m.setObjective(gamma * 0.5 * np.dot(theta, theta), GRB.MINIMIZE)

    for i in range(len(data_set)):
        item = data_set[i]
        m.addConstr(item[2] * (np.dot(theta, [item[0], item[1]])) >= 1, name=("cons%d" % i))

    m.update()
    m.write('Hw1_primal.lp')
    m.printStats()
    m.optimize()

    results = {}
    if m.Status == GRB.OPTIMAL:
        results['theta'] = theta

    return results


def solve_dual(x, y, gamma):
    m = Model()

    a = [ m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="a0"),
          m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="a1") ]

    u = []
    N = len(x)
    for i in xrange(N):
        u.append(m.addVar(name=("u%d" % i)))

    v = []
    M = len(y)
    for i in xrange(M):
        v.append(m.addVar(name=("v%d" % i)))

    m.update()

    m.setObjective(a[0]*a[0] + a[1]*a[1] + gamma*(quicksum(u) + quicksum(v)))

    for i in xrange(N):
        xi = x[i]
        m.addConstr(a[0]*xi[0] + a[1]*xi[1] >= 1 - u[i], name=("x%d" % i))

    for i in xrange(M):
        yi = y[i]
        m.addConstr(a[0]*yi[0] + a[1]*yi[1] <= -(1 - v[i]), name=("y%d" % i))

    m.update()
    m.write('Hw1_dual.lp')
    m.optimize()

    results = {}
    if m.Status == GRB.OPTIMAL:
        results['a'] = [a[0].X, a[1].X]

    return results

def predict(test_data_set):
    print test_data_set


if __name__ == '__main__':
    DATASET = "A"
    train_path = os.getcwd() + "/data/" + DATASET + "/test.csv"
    test_path = os.getcwd() + "/data/" + DATASET + "/test.csv"
    data_dict = vectorise(train_path)
    data_set = sanitize(train_path)

    X = []
    Y = []
    for item in data_set:
        X.append([item[0], item[1]])
        Y.append(item[2])

    primal_dict = solve_primal(data_set, 1)
    print primal_dict
    # dual_dict = solve_dual(data_dict[1], data_dict[-1], 1)
    # print dual_dict
    # plot_dict(data_dict)

