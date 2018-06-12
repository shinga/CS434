# @Author: Arthur Shing
# @Date:   2018-04-12T17:23:28-07:00
# @Filename: prob2.py
# @Last modified by:   Arthur Shing
# @Last modified time: 2018-04-16T05:51:02-07:00
import matplotlib
matplotlib.use('Agg')

import numpy as np
import csv
import matplotlib.pyplot as plt
import time


def loadData(fileName):
    data = np.loadtxt(fileName, delimiter=",")
    s = data.shape
    x = data
    label = np.asmatrix(data[:,-1]).T
    x = np.delete(x, -1, 1) # strip the 4 vs 9 bit
    w = np.zeros((s[1]-1, 1))
    return (x, label, w, data)

def predict(x, w):
    # for row in x:
    #     for i in row:
    #         continue
    return
    # w is just zeros, so the first iteration should return 0.5

# def predict1(row, w):
#     yhat = 0
#     for i in range(len(row)-1):
#         yhat += w[i] * row[i]
#     return 1.0 / (1.0 + np.exp(-yhat))
#
def checkAccuracy(x):
    if x > 0.5 or x < -0.5:
        return 0
    else:
        return 1


def sigmoidFunct(w, x):
    y = 1/(1 + np.exp(-np.dot(w.T, x)))
    return y

# Online gradient descent
def train(data, learnRate, epoch, zeros, label):
    w = zeros
    acc = np.zeros((epoch,), dtype=float)
    for e in range(epoch):
        sumError = 0
        for row in range(len(data)):
            y = sigmoidFunct(w, data[row].reshape(256,1))
            error = label[row] - y
            sumError += error**2
            odds = y * (1-y)
            learnError = error * learnRate * odds

            w = np.add(w, np.multiply(learnError, data[row].reshape(256,1))) # this took me 5ever
            # why does the above work so much better than the one below?
            # for pixel in range(len(data[row])):
            #     w[pixel] = w[pixel] + (learnRate*error*odds*data[row][pixel])

        acc[e] = test(data, label, w)
        # print acc[e]
        print('epoch: %d, learn rate: %.6f, SSE: %.6f, accuracy: %.6f' % (e, learnRate, sumError, acc[e]))
    return (w, acc)


# Batch gradient descent
def trainBatch(data, learnRate, epoch, zeros, label):
    w = zeros
    acc = np.zeros((epoch,), dtype=float)
    for e in range(epoch):
        sumError = 0
        nabla = w
        for row in range(len(data)):
            y = sigmoidFunct(w, data[row].reshape(256,1))
            error = label[row] - y
            sumError += error**2
            odds = y * (1-y)
            learnError = error * learnRate * odds

            nabla = np.add(nabla, np.multiply(learnError, data[row].reshape(256,1))) # this took me 5ever

        acc[e] = test(data, label, w)
        w = nabla
        # print acc[e]
        print('epoch: %d, learn rate: %.6f, SSE: %.6f, accuracy: %.6f' % (e, learnRate, sumError, acc[e]))

    return (w, acc)


def test(x, l, coef):
    numberWrong = 0
    y = sigmoidFunct(coef, x.T)
    y = y.T
    for i in range(len(y)):
        y[i] = round(y[i])
        if y[i] != l[i]:
            numberWrong += 1.0
        else:
            pass

    return (1.0 - (numberWrong / len(y)))




def main():
    #stuff

    (x, l, w, data) = loadData("usps-4-9-train.csv")
    (xTest, lTest, wTest, dataTest) = loadData("usps-4-9-test.csv")
    # for row in data[780:790]:
    #     yhat = predict1(row, w[0:])
    #     what = sigmoidFunct(w[0:], row[:-1].reshape(256,1))
    #     print("Expected=%.3f, Predicted=%.3f [%d] || %.3f [%d]" % (row[-1], yhat, round(yhat), what, round(what)))
    learn = 0.00005
    # learn = 0.000001
    epoch = 400
    (coef, accuracy) = train(x, learn, epoch, w, l)
    (coefB, accuracyB) = trainBatch(x, 0.000001, epoch, w, l)
    print test(xTest, lTest, coef)
    print test(xTest, lTest, coefB)

    # TODO
    # print coef
    # print accuracy.shape
    plot(range(epoch), accuracy, "acctrain.png")
    plot(range(epoch), accuracyB, "acctrain.png")

    # plot(test, "test.png")
    # Equation: prediction g(z)
        # g(z) = 1 / (1 + e^-(z)) where z is linear model a + bx + cx etc
        # as linear model approaches -infinity, prediction = 0
        # as linear model approaches 0, prediction = 0.5
        # as linear model approaches infinity, prediction = 1

def plot(x, y, fileName):
    plt.plot(x, y)
    plt.savefig(fileName)
    return


# Basically imshow in matlab, turns numbers into a pic
def showImage(w, name):
    pic = plt.imshow(np.reshape(w, (16,16)).T)
    plt.savefig(name)
    return

if __name__ == "__main__":
    main()
