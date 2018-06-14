# @Author: Arthur Shing
# @Date:   2018-04-12T17:23:28-07:00
# @Filename: prob2.py
# @Last modified by:   Arthur Shing
# @Last modified time: 2018-06-13T22:55:47-07:00
import matplotlib
matplotlib.use('Agg')

import numpy as np
import csv
import matplotlib.pyplot as plt
import time


def readFile(fileName):
    try:
        try:
            data = np.loadtxt(fileName, delimiter=",", converters = {0: lambda s: float(s[11:13])}, usecols=(0,1,2,3,4,5,6,7,8))
        except:
            data = np.loadtxt(fileName, delimiter=",", usecols=(0,1,2,3,4,5,6,7,8))
        s = data.shape
        print str(s) + " loaded"
        try:
            labels = np.loadtxt(fileName, delimiter=",", usecols=(9))
            labels = labels.reshape((labels.shape[0], 1))
            w = np.zeros((s[1], 1))
            return (data, labels, w)
        except:
            return data
    except:
        data = np.loadtxt(fileName, delimiter=",")
        s = data.shape
        print str(s) + " loaded"
        return data
    return data


def normalize(data):
    newData = data.T
    for column in range(newData.shape[0]):
        small = newData[column][0]
        large = newData[column][0]
        if (column != 0):
            for val in newData[column]:
                if val < small:
                    small = val
                elif val > large:
                    large = val
            for newval in range(newData[column].shape[0]):
                irange = large - small
                diff = newData[column][newval] - small
                newData[column][newval] = diff / irange
        else:
            continue
    return newData.T

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
            y = sigmoidFunct(w, data[row].reshape(63,1))
            error = label[row] - y
            sumError += error**2
            odds = y * (1-y)
            learnError = error * learnRate * odds

            w = np.add(w, np.multiply(learnError, data[row].reshape(63,1))) # this took me 5ever
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
            y = sigmoidFunct(w, data[row].reshape(63,1))
            error = label[row] - y
            sumError += error**2
            odds = y * (1-y)
            learnError = error * learnRate * odds

            nabla = np.add(nabla, np.multiply(learnError, data[row].reshape(63,1))) # this took me 5ever

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

def predict(x, coef, roundamount):

    y = sigmoidFunct(coef, x.T)
    y = y.T
    y = np.reshape(y, (y.shape[0],))
    result = []

    for i in range(len(y)):
        rounded = round(y[i] + roundamount)
        result.append((y[i], rounded))
    return result

def writepredictions(results, filename):
    with open('./predictions/' + filename, 'wb') as csvfile:
        spamwriter = csv.writer(csvfile)
        for row in results:
            spamwriter.writerow(row)

# def loadtestdata(fileName):
#     data = np.loadtxt(fileName, delimiter=",")
#     results = []
#     instances = []
#     for row in data:
#         instance = np.reshape(row, (63,7)).T
#         results.append(instance)
#         instances.append(instance[-1])
#     return (np.asarray(results), np.asarray(instances))

def loadtestdata(fileName):
    data = np.loadtxt(fileName, delimiter=",")
    # data = data.reshape(1,63)
    # results = []
    # instances = []
    # for row in data:
    #     instance = np.reshape(row, (9,7)).T
    #     results.append(instance)
    #     instances.append(instance[-1])
    return data
    # return (np.asarray(results), np.asarray(instances))

def findpos(data, l):
    positiveData = np.zeros((1, data.shape[1]*7))
    posl = []
    for i in range(data.shape[0]):
        if i > 7: #make sure there is 7 data points to grab from
            if l[i] == 1: # If data is a positive instance
                newData = np.zeros((1, data.shape[1]))
                newData = np.concatenate((newData, data[i-6].reshape(1,data.shape[1])), axis=0)
                newData = np.concatenate((newData, data[i-5].reshape(1,data.shape[1])), axis=0)
                newData = np.concatenate((newData, data[i-4].reshape(1,data.shape[1])), axis=0)
                newData = np.concatenate((newData, data[i-3].reshape(1,data.shape[1])), axis=0)
                newData = np.concatenate((newData, data[i-2].reshape(1,data.shape[1])), axis=0)
                newData = np.concatenate((newData, data[i-1].reshape(1,data.shape[1])), axis=0)
                newData = np.concatenate((newData, data[i].reshape(1,data.shape[1])), axis=0)
                newData = np.delete(newData, 0, 0)
                newData = newData.T
                newData = newData.reshape(1,(newData.shape[0])*7)
                positiveData = np.concatenate((positiveData, newData), axis=0)
                posl.append(l[i])
    positiveData = np.delete(positiveData, 0,0)
    return (positiveData, np.asarray(posl))




def findrand(data, l):
    plaindata = np.zeros((1, data.shape[1]*7))
    posl = []
    for i in range(7, data.shape[0], 7):
        if i > 7: #make sure there is 7 data points to grab from
            if (l[i] == 0 and
                l[i-1] == 0 and
                l[i-2] == 0 and
                l[i-3] == 0 and
                l[i-4] == 0 and
                l[i-5] == 0 and
                l[i-6] == 0): # If data is all negative instance
                newData = np.zeros((1, data.shape[1]))
                newData = np.concatenate((newData, data[i-6].reshape(1,data.shape[1])), axis=0)
                newData = np.concatenate((newData, data[i-5].reshape(1,data.shape[1])), axis=0)
                newData = np.concatenate((newData, data[i-4].reshape(1,data.shape[1])), axis=0)
                newData = np.concatenate((newData, data[i-3].reshape(1,data.shape[1])), axis=0)
                newData = np.concatenate((newData, data[i-2].reshape(1,data.shape[1])), axis=0)
                newData = np.concatenate((newData, data[i-1].reshape(1,data.shape[1])), axis=0)
                newData = np.concatenate((newData, data[i].reshape(1,data.shape[1])), axis=0)
                newData = np.delete(newData, 0, 0)
                newData = newData.T
                newData = newData.reshape(1,(newData.shape[0])*7)
                plaindata = np.concatenate((plaindata, newData), axis=0)
                posl.append(l[i])
    plaindata = np.delete(plaindata, 0,0)
    return (plaindata, np.asarray(posl))

def main():
    #stuff
    # gentestinst1 = readFile("sampleinstance_1.csv")
    # gentestinst2 = readFile("sampleinstance_2.csv")
    # gentestinst3 = readFile("sampleinstance_3.csv")
    # gentestinst4 = readFile("sampleinstance_4.csv")
    # gentestinst5 = readFile("sampleinstance_5.csv")
    # gentestinst = np.concatenate((gentestinst1, gentestinst2, gentestinst3, gentestinst4, gentestinst5), axis=0)
    # (gentest,gentestinst) = loadtestdata("general_test_instances.csv")

    sub2test = loadtestdata("subject2_instances.csv")
    sub2test = normalize(sub2test)
    sub7test = loadtestdata("subject7_instances.csv")
    sub7test = normalize(sub7test)
    # gentestinst = loadtestdata("sampleinstance_2.csv")
    # print gentest.shape
    # print gentestinst.shape
    (train2, label2, w2) = readFile("Subject_2_part1.csv")
    (train7, label7, w7) = readFile("Subject_7_part1.csv")


    w2 = np.zeros((63, 1))
    w7 = np.zeros((63, 1))

    # dataTrain = np.hstack((trainIndex.reshape(trainIndex.shape[0], 1), dataTrain))
    x2 = normalize(train2)
    x7 = normalize(train7)
    print x2[0]
    print x2[1]
    print x7[0]
    print x7[1]


    (pos, posl) = findpos(x2, label2)
    (neg, negl) = findrand(x2, label2)
    x = np.concatenate((pos, neg), axis=0)
    l = np.concatenate((posl, negl), axis=0)
    print x.shape


    #
    # (x, l, w, data) = loadData("usps-4-9-train.csv")
    # (xTest, lTest, wTest, dataTest) = loadData("usps-4-9-test.csv")
    # # for row in data[790:790]:
    # #     yhat = predict1(row, w[0:])
    # #     what = sigmoidFunct(w[0:], row[:-1].reshape(9,1))
    # #     print("Expected=%.3f, Predicted=%.3f [%d] || %.3f [%d]" % (row[-1], yhat, round(yhat), what, round(what)))
    learn = 0.000005
    # learn = 0.000001
    epoch = 20
    (coef, accuracy) = train(x, learn, epoch, w2, l)
    (coefB, accuracyB) = trainBatch(x, 0.000001, epoch, w2, l)
    # print predict(gentestinst, coef)
    # print predict(gentestinst, coefB)
    writepredictions(predict(sub2test, coef, 0),"individual1_pred1.csv")

    (pos, posl) = findpos(x7, label7)
    (neg, negl) = findrand(x7, label7)
    x = np.concatenate((pos, neg), axis=0)
    l = np.concatenate((posl, negl), axis=0)
    print x.shape

    (coef, accuracy) = train(x, learn, epoch, w7, l)
    (coefB, accuracyB) = trainBatch(x, 0.000001, epoch, w7, l)
    # print predict(gentestinst, coef)
    # print predict(gentestinst, coefB)
    writepredictions(predict(sub7test, coef, 0),"individual2_pred1.csv")








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
