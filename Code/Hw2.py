## Adding RBF kernels
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

def rbf(va, vb, sigma):
    return exp(-sigma * linalg.norm(va - vb) ** 2)


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


def plot_dict(data_dict, theta_X, theta_Y, gamma, dataset):
    colors = list("rb")

    for arr in data_dict.values():
        (x, y) = arr.T
        plt.scatter(x, y, color=colors.pop())

    plt.legend(data_dict.keys())
    x_points = range(-4, 4)
    y_points = []
    y_points_neg = []
    y_points_pos = []
    m = theta_X/theta_Y
    b = 1/math.sqrt(theta_X * theta_X + theta_Y * theta_Y)
    for x_point in x_points:
        y_points.append(-m*x_point)
        y_points_neg.append(-m*x_point - b)
        y_points_pos.append(-m*x_point + b)

    plt.plot(y_points, x_points, 'g')
    plt.title('Gamma ' + str(gamma) + " Dataset " + dataset)
    plt.plot(y_points_neg, x_points,linestyle='dotted', color='green')
    plt.plot(y_points_pos, x_points, linestyle='dotted', color='green')

    plt.show()



def solve_primal(data_set, gamma):
    m = Model()

    theta = [m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="theta_X"),
             m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="theta_Y")]
    # theta0 = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="theta0")

    u = []
    for i in range(len(data_set)):
        u.append(m.addVar(name=("u%d" % i)))

    m.setObjective((theta[0] * theta[0] + theta[1] * theta[1]) + gamma * quicksum(u), GRB.MINIMIZE)


    # shuffle(data_set)
    # print type(x)



    for i in range(len(data_set)):
        item = data_set[i]
        # print theta[0] *item[0] + theta[1] * item[1]
        m.addConstr((item[2] * (theta[0] * item[0] + theta[1] * item[1]) >= 1 - u[i]), name=("cons%d" % i))

    m.write('Hw1_primal.lp')
    # m.printStats()
    m.optimize()

    results = dict()
    results['theta_X'] = theta[0].X
    results['theta_Y'] = theta[1].X
    # results['theta0'] = theta0.X

    return results



def solve_dual(X, Y, gamma, sigma):
    m = Model("qp")

    a = []

    u = []
    N = len(X)
    for i in xrange(N):
        a.append(m.addVar(lb=0, ub=1/gamma, name=("a%d" % i)))

    print len(a)

    a = np.asarray(a)

    a_sum = QuadExpr()
    # kerX = [[0 for x in range(N)] for y in range(N)]
    # for i in xrange(N):
    #     for j in xrange(N):
    #         kerX[i][j] = np.dot(X[i], X[j])

    for i in xrange(N):
        # print('i:', i)
        for j in xrange(N):
            # print (i, j)
            a_sum.add(a[i]*a[j] * rbf(X[i], X[j], sigma) * Y[i] * Y[j])

    m.setObjective(quicksum(a) - 0.5 * a_sum , GRB.MAXIMIZE)

    print('Obj set')

    sum = 0
    for i in xrange(N):
        sum = sum + Y[i]*a[i]
    print type(sum)
    m.addConstr(sum == 0)

    m.update()
    m.write('Hw1_dual.lp')
    m.optimize()


    results = {}
    theta_X = 0;
    theta_Y = 0


    if m.Status == GRB.OPTIMAL:
        for i in xrange(N):
            theta_X = theta_X + X[i][0] * a[i].X * Y[i]
            theta_Y = theta_Y + X[i][1] * a[i].X * Y[i]
        results['theta'] = [theta_X, theta_Y]
        print results

    return results



def sign(theta_X, theta_Y, item, sigma):
    if rbf(np.asarray([theta_X, theta_Y]), np.asarray([item[0], item[1]]), sigma) > 0:
        return 1
    else: return -1


def predict(test_data_set, theta_X, theta_Y, sigma):
    hits = 0

    for item in test_data_set:
        print (sign(theta_X, theta_Y, item, sigma))
        if (sign(theta_X, theta_Y, item, sigma)) == item[2]:
            hits = hits + 1

    print "Hits :- " + str(hits)
    print "Misses :- " + str(len(test_data_set) - hits)
    print "Total :- " + str(len(test_data_set))
    print "Accuracy is " + str(float(hits)/float(len(test_data_set)) * 100) + "%"


def xfrange(start, stop, step):
    i = 0
    while start + i * step < stop:
        yield start + i * step
        i += 1




if __name__ == '__main__':
    DATASETS = ["C"]
    gammas = [1]

    #A use sigma = 2.19
    #B use sigma = 0.92
    #C use sigma = 4.2

    sigmas = [4.2]

    for DATASET in DATASETS:
        for gamma in gammas:
            for sigma in sigmas:
                train_path = os.getcwd() + "/data/" + DATASET + "/train.csv"
                test_path = os.getcwd() + "/data/" + DATASET + "/test.csv"
                train_data_dict = vectorise(train_path)
                train_data_set = sanitize(train_path)
                test_data_set = sanitize(test_path)
                test_data_dict = vectorise(test_path)

                X_train = []
                Y_train = []
                for item in train_data_set:
                    X_train.append([item[0], item[1]])
                    Y_train.append(item[2])

                X_test = []
                Y_test = []
                for item in train_data_set:
                    X_test.append([item[0], item[1]])
                    Y_test.append(item[2])





                dual_dict = solve_dual(np.asarray(X_train), np.asarray(Y_train), gamma, sigma)
                print "gamma is " + str(gamma)
                print "sigma is " + str(sigma)

                print "For dataset " + DATASET
                print "Values from dual form " + str(dual_dict)

                predict(test_data_set, dual_dict['theta'][0], dual_dict['theta'][1], sigma)

                # primal_dict = solve_primal(train_data_set, gamma)
                # print "Values from primal form " + str(primal_dict)
                # predict(test_data_set, primal_dict['theta_X'], primal_dict['theta_Y'])

                # plot_dict(train_data_dict, primal_dict['theta_X'], primal_dict['theta_Y'], gamma, DATASET)

