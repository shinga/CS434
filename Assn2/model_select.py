# @Author: Arthur Shing
# @Date:   2018-04-25T17:07:51-07:00
# @Filename: model_select.py
# @Last modified by:   Arthur Shing
# @Last modified time: 2018-04-29T16:40:26-07:00

import matplotlib
matplotlib.use('Agg')


import numpy as np
import matplotlib.pyplot as plt
import math
from operator import itemgetter
np.set_printoptions(threshold=np.nan)


def readFile(fileName):
    data = np.loadtxt(fileName, delimiter=",")
    s = data.shape
    print str(s) + " loaded"
    return data

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
def knn(dataTrain, dataTest, k):
    predicted = 0
    (correctVal, testVal, eud) = calcEuclDist(dataTrain, dataTest)
    neighbors = getNearestNeighbors(eud, correctVal, k)
    neVal = []
    for j in neighbors:
        neVal.append(j[1])
    if(sum(neVal) > 0):
        predicted = 1
    elif (sum(neVal) < 0):
        predicted = -1
    else:
        raise ValueError("k needs to be odd")
    return predicted

# dataTrain is the matrix of all traininig data
# dataTest is a single data array
def LOOXV(dataTrain, dataTest, k):
    predicted = 0
    (correctVal, testVal, eud) = calcEuclDist(dataTrain, dataTest)
    neighbors = getNearestNeighborsLOOXV(eud, correctVal, k)
    neVal = []
    for j in neighbors:
        neVal.append(j[1])
    if(sum(neVal) > 0):
        predicted = 1
    elif (sum(neVal) < 0):
        predicted = -1
    else:
        raise ValueError("k needs to be odd")
    return predicted

def testError(dataTrain, dataTest, k):
    wrong = 0
    for i in dataTest:
        p = knn(dataTrain, i, k)
        if (i[0] != p):
            wrong += 1
    # testError = (wrong / (dataTest.shape[0]))
    return (float(wrong) / dataTest.shape[0])


def testErrorLOOXV(dataTrain, dataTest, k):
    wrong = 0
    for i in dataTest:
        p = LOOXV(dataTrain, i, k)
        if (i[0] != p):
            wrong += 1
    # testError = (wrong / (dataTest.shape[0]))
    return ((float(wrong) / dataTest.shape[0]))

def main():
    dataTrain = readFile("knn_train.csv")
    dataTest = readFile("knn_test.csv")
    normData = normalize(dataTrain)
    normDataTest = normalize(dataTest)
    # print normData[0]
    # print normData[1]
    # print test[0]
    # (correctVal, testVal, eud) = calcEuclDist(normData, normData[1])
    # (correctVal, testVal, eud) = calcEuclDist(normData, normDataTest[1])

    k = 3
    # test = getNearestNeighbors(eud, correctVal, k)
    # test = knn(normData, normDataTest[1], k)
    test = testError(normData, normData, k)
    print ("Training data with k = %d: %f" % (k, test))
    test = testErrorLOOXV(normData, normData, k)
    print ("Leave-one-out cross validation error with k = %d: %f" % (k, test))
    test = testError(normData, normDataTest, k)
    print ("Testing data with k = %d: %f" % (k, test))
    train = []
    looxv = []
    test = []
    for k in range(1, 53, 2):
        print ("Getting data for k=%d..." % k)
        train.append(testError(normData, normData, k))
        # print ("Training data with k = %d: %f" % (k, test))
        looxv.append(testErrorLOOXV(normData, normData, k))
        # print ("Leave-one-out cross validation error with k = %d: %f" % (k, test))
        test.append(testError(normData, normDataTest, k))
        # print ("Testing data with k = %d: %f" % (k, test))
    plt.plot(range(1,53,2), train, 'bx-', label="Training Error")
    plt.plot(range(1,53,2), looxv, 'gx-', label="Leave-One-Out Error")
    plt.plot(range(1,53,2), test, 'rx-', label="Testing Error")
    plt.xlabel('k')
    plt.ylabel('Error rate')
    plt.savefig("errors.png")






if __name__ == "__main__":
    main()
