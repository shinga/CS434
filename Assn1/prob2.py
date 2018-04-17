# @Author: Arthur Shing & Monica Sek
# @Date:   2018-04-12T17:23:28-07:00
# @Filename: prob2.py
# @Last modified by:   Arthur Shing
# @Last modified time: 2018-04-16T19:06:16-07:00
import matplotlib
matplotlib.use('Agg')

import numpy as np
import csv
import matplotlib.pyplot as plt
import time

def main():
    #stuff
    problem1()

def loadData(fileName):
    data = np.loadtxt(fileName, delimiter=",")
    s = data.shape
    x = data
    label = np.asmatrix(data[:,-1]).T
    x = np.delete(x, -1, 1) # strip the 4 vs 9 bit
    w = np.zeros((s[1]-1, 1))
    return (x, label, w, data)


def sigmoidFunct(w, x):
    y = 1/(1 + np.exp(-np.dot(w.T, x)))

    # y = np.add(y, regularizator)

    return y


# Batch gradient descent
# data = training dataset
# epoch = number of iterations
# zeros should be an array of zeros. Unnecessary and I should fix it later.
# label = array of 'is training data 4 or 9?' (0 is 4, 1 is 9)
# xTest and lTest is the data and label for the test file
# Returns:
# w = the coefficients for the sigmoid function
# acc = array of the accuracy of each iteration on the training dataset
# accTest = array of the accuracy of each iteration on the test dataset
# l = lambda
def trainBatch(data, learnRate, epoch, zeros, label, xTest, lTest, l):
    w = zeros
    acc = np.zeros((epoch,), dtype=float)
    accTest = np.zeros((epoch,), dtype=float)
    for e in range(epoch):
        sumError = 0
        nabla = w
        for row in range(len(data)):
            y = sigmoidFunct(w, data[row].reshape(256,1))
            regularizator = (0.5 * np.dot(w.T, w) * l)

            error = label[row] - y
            sumError += error**2
            odds = y * (1-y)
            regError = error + regularizator
            learnError = regError * learnRate * odds

            nabla = np.add(nabla, np.multiply(learnError, data[row].reshape(256,1))) # this took me 5ever

        acc[e] = test(data, label, w)
        accTest[e] = test(xTest, lTest, w)
        w = nabla
        print('epoch: %d, learn rate: %.6f, SSE: %.6f, accuracy: %.6f, accuracyTest: %.6f' % (e, learnRate, sumError, acc[e], accTest[e]))
    return (w, acc, accTest)

# Tests multiple images (x is the test file data) over their actual numbers (l)
# Returns percent correct in decimal form (0 to 1)
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


def problem1():
    (x, l, w, data) = loadData("usps-4-9-train.csv")
    (xTest, lTest, wTest, dataTest) = loadData("usps-4-9-test.csv")

    # Learning Rate
    learn = 0.000001
    # Iterations
    epoch = 65

    la = 100

    # Accuracy = array of % correctly predicted in each iteration
    (coefB, accuracyB, accuracyTestB) = trainBatch(x, learn, epoch, w, l, xTest, lTest, la)

    # TODO:: Add legend and labels
    plot(range(epoch), accuracyB, "acctrain.png", "Training data")
    plot(range(epoch), accuracyTestB, "acctrain.png", "Testing data")


    # Part 3
    (coefB, accuracyB, accuracyTestB) = trainBatch(x, learn, epoch, w, l, xTest, lTest, 0.1)
    plot(range(epoch), accuracyB, "acctrain2.png", "Training data lambda=0.1")
    plot(range(epoch), accuracyTestB, "acctrain2.png", "Testing data lambda=0.1")
    (coefB, accuracyB, accuracyTestB) = trainBatch(x, learn, epoch, w, l, xTest, lTest, 1)
    plot(range(epoch), accuracyB, "acctrain2.png", "Training data lambda=1")
    plot(range(epoch), accuracyTestB, "acctrain2.png", "Testing data lambda=1")
    (coefB, accuracyB, accuracyTestB) = trainBatch(x, learn, epoch, w, l, xTest, lTest, 10)
    plot(range(epoch), accuracyB, "acctrain2.png", "Training data lambda=10")
    plot(range(epoch), accuracyTestB, "acctrain2.png", "Testing data lambda=10")
    (coefB, accuracyB, accuracyTestB) = trainBatch(x, learn, epoch, w, l, xTest, lTest, 100)
    plot(range(epoch), accuracyB, "acctrain2.png", "Training data lambda=100")
    plot(range(epoch), accuracyTestB, "acctrain2.png", "Testing data lambda=100")
    (coefB, accuracyB, accuracyTestB) = trainBatch(x, learn, epoch, w, l, xTest, lTest, 1000)
    plot(range(epoch), accuracyB, "acctrain2.png", "Training data lambda=1000")
    plot(range(epoch), accuracyTestB, "acctrain2.png", "Testing data lambda=1000")
    (coefB, accuracyB, accuracyTestB) = trainBatch(x, learn, epoch, w, l, xTest, lTest, 10000)
    plot(range(epoch), accuracyB, "acctrain2.png", "Training data lambda=10000")
    plot(range(epoch), accuracyTestB, "acctrain2.png", "Testing data lambda=10000")



    return

# Plots into a png file
def plot(x, y, fileName, labelName):
    plt.plot(x, y, label=labelName)
    plt.legend(loc=4)
    plt.savefig(fileName)
    return


# Basically imshow in matlab, turns numbers into a pic
def showImage(w, name):
    pic = plt.imshow(np.reshape(w, (16,16)).T)
    plt.savefig(name)
    return

if __name__ == "__main__":
    main()
