# @Author: Arthur Shing
# @Date:   2018-04-25T17:07:51-07:00
# @Filename: model_select.py
# @Last modified by:   Arthur Shing
# @Last modified time: 2018-06-13T22:48:07-07:00

import matplotlib
matplotlib.use('Agg')


import numpy as np
import matplotlib.pyplot as plt
import math
import csv
from operator import itemgetter
np.set_printoptions(threshold=np.nan)


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

# one = training example
# two = test
# oneClass = training example class
# twoClass = test case class (as reported)
def euclideanDistance(one, two):
    # one and two are arrays
    oneClass = one[0]
    twoClass = two[0]
    dist = 0
    distAdd = np.subtract(two, one)
    distAdd = np.square(distAdd)
    distAdd = distAdd[1:]
    distAdd = np.sum(distAdd)
    distAdd = np.sqrt(distAdd)
    return (oneClass, twoClass, distAdd)

def calcEuclDist(trainMatrix, test):
    if (trainMatrix.shape[1] == test.shape[0]):
        ans = np.apply_along_axis(euclideanDistance, 1, arr=trainMatrix, two=test)
        correctVal = ans.T[0]
        testVal = ans.T[1]
        ed = ans.T[2]
    return (correctVal, testVal, ed)


def getNearestNeighbors(eud, correctVal, k):
    indexList = []
    for i in range(eud.shape[0]):
        indexList.append((i, correctVal[i], eud[i]))
    sortedIndex = sorted(indexList, key=itemgetter(2))

    # if (sortedIndex[0][2] == 0):
    #     return sortedIndex[1:(k+1)]
    # else:
    return sortedIndex[0:k]

def getNearestNeighborsLOOXV(eud, correctVal, k):
    indexList = []
    for i in range(eud.shape[0]):
        indexList.append((i, correctVal[i], eud[i]))
    sortedIndex = sorted(indexList, key=itemgetter(2))

    if (sortedIndex[0][2] == 0):
        return sortedIndex[1:(k+1)]
    else:
        return sortedIndex[0:k]

# dataTrain is the matrix of all traininig data
# dataTest is a single data array
def knn(dataTrain, dataTest, k, l):
    predicted = 0
    (correctVal, testVal, eud) = calcEuclDist(dataTrain, dataTest)
    neighbors = getNearestNeighbors(eud, correctVal, k)
    neVal = []
    for j in neighbors:
        # print neighbors
        neVal.append(l[j[0]])

    print sum(neVal)
    if(sum(neVal) > 0):
        predicted = np.mean(neVal)

    return predicted


def testError(dataTrain, dataTest, k, l):
    wrong = 0
    iterator = 0
    result = []
    for i in range(dataTest.shape[0]):
        print i, k
        p = knn(dataTrain, dataTest[i], k, l)
        rounded = lambda x: 1 if x >= 0.1 else 0
        result.append((p, rounded(p)))

    return result
    #
    #
    #
    #     if (l[i] != round(p)):
    #         wrong += 1
    #     iterator += 1
    #     print(iterator)
    # # testError = (wrong / (dataTest.shape[0]))
    # return (float(wrong) / dataTest.shape[0])


def main():

    sub2test = loadtestdata("subject2_instances.csv")
    sub2test = normalize(sub2test)
    sub7test = loadtestdata("subject7_instances.csv")
    sub7test = normalize(sub7test)
    # (gentestinst, label5, w1) = readFile("Subject_2_part1.csv")

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

    k = 20
    test = testError(x, sub2test, k, l)
    writepredictions(test,"individual1_pred2.csv")

    (pos, posl) = findpos(x7, label7)
    (neg, negl) = findrand(x7, label7)
    x = np.concatenate((pos, neg), axis=0)
    l = np.concatenate((posl, negl), axis=0)
    print x.shape

    k = 20
    test = testError(x, sub2test, k, l)
    writepredictions(test,"individual2_pred2.csv")
    # gentestinst = x
    # # print test[0]
    # # (correctVal, testVal, eud) = calcEuclDist(normData, normData[1])
    # # print correctVal.shape

    # # (correctVal, testVal, eud) = calcEuclDist(normData, normDataTest[1])
    #
    # test = getNearestNeighbors(eud, correctVal, k)
    # test = knn(normData, normDataTest[1], k)


    # test = testError(x, x, k)
    # print ("Training data with k = %d: %f" % (k, test))




    # for k in range(1, 53, 2):
    #     print ("Getting data for k=%d..." % k)
    #     train.append(testError(normData, normData, k))
    #     # print ("Training data with k = %d: %f" % (k, test))
    #     looxv.append(testErrorLOOXV(normData, normData, k))
    #     # print ("Leave-one-out cross validation error with k = %d: %f" % (k, test))
    #     test.append(testError(normData, normDataTest, k))
    #     # print ("Testing data with k = %d: %f" % (k, test))
    # plt.plot(range(1,53,2), train, 'bx-', label="Training Error")
    # plt.plot(range(1,53,2), looxv, 'gx-', label="Leave-One-Out Error")
    # plt.plot(range(1,53,2), test, 'rx-', label="Testing Error")
    # plt.xlabel('k')
    # plt.ylabel('Error rate')
    # plt.legend(loc=4, prop={'size':10})
    # plt.savefig("errors.png")






if __name__ == "__main__":
    main()
