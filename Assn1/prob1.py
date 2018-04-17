# @Author: Arthur Shing & Monica Sek
# @Date:   2018-04-12T19:20:42-07:00
# @Filename: prob1.py
# @Last modified by:   Arthur Shing
# @Last modified time: 2018-04-16T20:23:56-07:00

import numpy
import matplotlib as mpl
numpy.set_printoptions(precision=4, threshold=numpy.inf)

# Returns: X is features data, Y is housing price
def getValues(fileName):
    # Load files into X and Y matrices
    data = numpy.loadtxt(fileName)
    s = data.shape
    ones = numpy.ones((s[0], 1))
    x = numpy.hstack((ones, data))
    y = numpy.asmatrix(data[:,13]).T
    # Remove house price from X after moving it to Y
    x = numpy.delete(x, 14, 1)
    return(x, y)

def addRandFeature(d, data, y):
    x = numpy.concatenate((data, y), axis=1)
    # x = data
    xAvg = numpy.mean(x, axis=0)
    xStd = numpy.std(x, axis=0)
    xAvg = numpy.asarray(xAvg)
    xStd = numpy.asarray(xStd)

    randFeat = numpy.zeros((d, x.shape[1]))
    for n in range(d):
        for i in range(14):
            # randFeat[n][i] = xAvg[0][i] + (xStd[0][i] * numpy.random.standard_normal(1))
            randFeat[n][i] =  numpy.random.normal(xAvg[0][i], xStd[0][i])
            # jifew =  numpy.random.standard_normal(1)
            # print xStd[0][i] *
    x = numpy.vstack((randFeat, x))

    y = numpy.asmatrix(x[:,13]).T
    x = numpy.delete(x, 14, 1)
    return (x, y.T)



def weight(x, y):
    # (XT * X)^-1
    xTx = numpy.dot(x.T, x)
    # print "Hello"
    # print xTx.shape
    xTxInv = numpy.mat(xTx).I

    # XT * Y
    xTy = numpy.dot(x.T, y)
    # Put it together: (XT * T)^-1 * XT*Y
    w = numpy.dot(xTxInv, xTy)
    return w

def sse(x, y, w):
    # Get SSE (product of X and w, squared)
    fx = numpy.dot(x, w)
    e = y - fx
    eTe = numpy.dot(e.T, e)
    return eTe

def getValuesNoDummy(fileName):
    # Load files into X and Y matrices
    data = numpy.loadtxt(fileName)
    s = data.shape
    x = data
    y = numpy.asmatrix(data[:,13]).T
    # Remove house price from X after moving it to Y
    x = numpy.delete(x, 13, 1)
    return(x, y)

def main():
    # Problem 1
    print "Problem 1."
    (xTrain, yTrain) = getValues("housing_train.txt")
    wTrain = weight(xTrain, yTrain)
    sseTrain = sse(xTrain, yTrain, wTrain)
    print "Weight vector:"
    print wTrain.reshape(1,14)

    (xTest, yTest) = getValues("housing_test.txt")
    # wTest = weight(xTest, yTest) oops we use weighted vector for train
    sseTest = sse(xTest, yTest, wTrain)

    print "Training ASE:"
    print sseTrain
    print "Testing ASE:"
    print sseTest

    (xTrainND, yTrainND) = getValuesNoDummy("housing_train.txt")
    wTrainND = weight(xTrainND, yTrainND)
    sseTrainND = sse(xTrainND, yTrainND, wTrainND)

    (xTestND, yTestND) = getValuesNoDummy("housing_test.txt")
    sseTestND = sse(xTestND, yTestND, wTrainND)

    print "Training ASE with no dummy:"
    print sseTrainND
    print "Testing ASE with no dummy:"
    print sseTestND

    # Problem 4
    for r in [2,4,6,8,10]:
        (xRand, yRand) = addRandFeature(r, xTrain, yTrain)
    wRand = weight(xRand, yRand)
    sseRand = sse(xTrain, yTrain, wRand)
    sseRandTest = sse(xTest, yTest, wRand)

    print "Problem 4."
    print "Weight vector:"
    print wRand.reshape(1,14)
    print "Training ASE:"
    print sseRand
    print "Testing ASE:"
    print sseRandTest


if __name__ == "__main__":

    main()
