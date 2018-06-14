# @Author: Arthur Shing & Monica Sek
# @Date:   2018-04-12T19:20:42-07:00
# @Filename: prob1.py
# @Last modified by:   Arthur Shing
# @Last modified time: 2018-06-13T23:32:21-07:00
import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import math
import csv
np.set_printoptions(precision=4, threshold=np.inf)



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


# Returns: X is features data, Y is housing price
def getValues(fileName):
    # Load files into X and Y matrices
    data = np.loadtxt(fileName)
    s = data.shape
    ones = np.ones((s[0], 1))
    x = np.hstack((ones, data))
    y = np.asmatrix(data[:,13]).T
    # Remove house price from X after moving it to Y
    x = np.delete(x, 14, 1)
    return(x, y)

def addRandFeature(d, data, y):
    x = np.concatenate((data, y), axis=1)
    # x = data
    xAvg = np.mean(x, axis=0)
    xStd = np.std(x, axis=0)
    xAvg = np.asarray(xAvg)
    xStd = np.asarray(xStd)

    randFeat = np.zeros((d, x.shape[1]))
    for n in range(d):
        for i in range(14):
            # randFeat[n][i] = xAvg[0][i] + (xStd[0][i] * np.random.standard_normal(1))
            randFeat[n][i] =  np.random.normal(xAvg[0][i], xStd[0][i])
            # jifew =  np.random.standard_normal(1)
            # print xStd[0][i] *
    x = np.vstack((randFeat, x))

    y = np.asmatrix(x[:,13]).T
    x = np.delete(x, 14, 1)
    return (x, y.T)



def weight(x, y):
    # (XT * X)^-1
    xTx = np.dot(x.T, x)
    # print "Hello"
    # print xTx.shape
    xTxInv = np.mat(xTx).I

    # XT * Y
    xTy = np.dot(x.T, y)
    # Put it together: (XT * T)^-1 * XT*Y
    w = np.dot(xTxInv, xTy)
    return w

def sse(x, w):
    # Get SSE (product of X and w, squared)
    fx = np.dot(x, w)
    print fx.shape
    results = []
    rounded = lambda x: 1 if x > 0 else 0
    for i in fx:
        try:
            ix = i.tolist()
            ix = [item for items in ix for item in items]
            results.append((ix[0], rounded(ix[0])))
        except:
            results.append((i[0], rounded(i)))
    return results

def getValuesNoDummy(fileName):
    # Load files into X and Y matrices
    data = np.loadtxt(fileName)
    s = data.shape
    x = data
    y = np.asmatrix(data[:,13]).T
    # Remove house price from X after moving it to Y
    x = np.delete(x, 13, 1)
    return(x, y)

def main():

    gentestinst = loadtestdata("general_test_instances.csv")
    gentestinst = normalize(gentestinst)

    # (gentestinst, label5, w1) = readFile("Subject_2_part1.csv")

    # gentestinst = loadtestdata("sampleinstance_2.csv")
    # print gentest.shape
    # print gentestinst.shape
    (train1, label1, w1) = readFile("Subject_1.csv")
    (train2, label2, w2) = readFile("Subject_4.csv")
    (train3, label3, w3) = readFile("Subject_6.csv")
    (train4, label4, w4) = readFile("Subject_9.csv")
    dataTrain = np.concatenate((train1, train2, train3, train4), axis=0)
    l = np.concatenate((label1, label2, label3, label4), axis=0)
    w = np.zeros((63, 1))
    print dataTrain.shape
    print l.shape
    print w.shape
    lt1 = readFile("list_1.csv")
    lt2 = readFile("list_4.csv")
    lt3 = readFile("list_6.csv")
    lt4 = readFile("list_9.csv")
    trainIndex = np.concatenate((lt1, lt2, lt3, lt4), axis=0)
    # dataTrain = np.hstack((trainIndex.reshape(trainIndex.shape[0], 1), dataTrain))
    x = normalize(dataTrain)
    print x[0]
    print x[1]


    (pos, posl) = findpos(x, l)
    (neg, negl) = findrand(x, l)
    x = np.concatenate((pos, neg), axis=0)
    l = np.concatenate((posl, negl), axis=0)
    print x.shape
    # gentestinst = x
    weightvector = weight(x, l)
    # print weightvector
    # print weightvector.shape
    sseTrain = sse(x, weightvector)
    sseTest = sse(gentestinst, weightvector)
    writepredictions(sseTest,"general_pred3.csv")


    #
    #
    # # Problem 1
    # print "Problem 1."
    # (xTrain, yTrain) = getValues("housing_train.txt")
    # wTrain = weight(xTrain, yTrain)
    # sseTrain = sse(xTrain, yTrain, wTrain)
    # print wTrain.reshape(1,14)
    #
    # (xTest, yTest) = getValues("housing_test.txt")
    # # wTest = weight(xTest, yTest) oops we use weighted vector for train
    # sseTest = sse(xTest, yTest, wTrain)
    #
    # print "Training ASE:"
    # print sseTrain
    # print "Testing ASE:"
    # print sseTest
    #
    # (xTrainND, yTrainND) = getValuesNoDummy("housing_train.txt")
    # wTrainND = weight(xTrainND, yTrainND)
    # sseTrainND = sse(xTrainND, yTrainND, wTrainND)
    #
    # (xTestND, yTestND) = getValuesNoDummy("housing_test.txt")
    # sseTestND = sse(xTestND, yTestND, wTrainND)
    #
    # print "Training ASE with no dummy:"
    # print sseTrainND
    # print "Testing ASE with no dummy:"
    # print sseTestND
    #
    # # Problem 4
    # sseRand = np.zeros(5)
    # sseRandTest = np.zeros(5)
    # for r in range(5):
    #     (xRand, yRand) = addRandFeature((r*2), xTrain, yTrain)
    #     wRand = weight(xRand, yRand)
    #     sseRand[r] = np.asarray(sse(xTrain, yTrain, wRand))
    #     sseRandTest[r] = np.asarray(sse(xTest, yTest, wRand))
    # plt.subplot(2, 1, 1)
    # plt.title('Random Features ')
    # plt.ylabel('ASE')
    #
    # plot(range(0,10,2), sseRand.flatten(), "randfeatures.png", "Training")
    # plt.subplot(2, 1, 2)
    # plt.ylabel('ASE')
    # plt.xlabel('Number of Features')
    # plot(range(0,10,2), sseRandTest.flatten(), "randfeatures.png", "Testing")
    #
    # print "Problem 4."
    # print "Weight vector:"
    # print wRand.reshape(1,14)
    # print "Training ASE:"
    # print sseRand[0]
    # print "Testing ASE:"
    # print sseRandTest[0]

    return


def plot(x, y, fileName, labelName):
    plt.plot(x, y, label=labelName)
    plt.legend(loc=4, prop={'size':10})
    plt.savefig(fileName)
    return


if __name__ == "__main__":

    main()
