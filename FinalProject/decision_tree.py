# @Author: Arthur Shing, Monica Sek
# @Date:   2018-04-29T16:41:18-07:00
# @Filename: decision_tree.py
# @Last modified by:   Arthur Shing
# @Last modified time: 2018-06-09T20:24:06-07:00

import matplotlib
matplotlib.use('Agg')


import numpy as np
import matplotlib.pyplot as plt
import math
np.set_printoptions(threshold=np.nan)


class Node(object):
    def __init__(self):
        self.left = None
        self.right = None
        self.threshold = None
        self.column = None
        self.depth = None
        self.leftError = None
        self.rightError = None
        self.data = None

    def getVals(self):
        return (self.threshold, self.column)

    def isLeaf(self):
        return self.right is None and self.left is None

    def setVals(self, thresh, col):
        self.threshold = thresh
        self.column = col

    def printNode(self):
        print "Depth: %d" % self.depth
        print self.column, self.threshold
        print "error: "
        print self.leftError, self.rightError



def readFile(fileName):
    try:
        data = np.loadtxt(fileName, delimiter=",", usecols=(1,2,3,4,5,6,7,8,9))
    except:
        data = np.loadtxt(fileName, delimiter=",")
    s = data.shape
    print str(s) + " loaded"
    return data

# Calculate entropy given probabilities of +1 (p1) and negative 1 (pn1)
def entropy(p1, pn1):
    if(p1 == 0 or pn1 == 0):
        return 0
    l2 = np.log2(p1)
    nl2 = np.log2(pn1)
    Hs = -((p1*l2) + (pn1*nl2))
    # print Hs
    return Hs


# Count the number of 1s in a column
# Return probability of 1 and -1 in the column (p1, pn1)
def countOnes(column):
    numOnes = 0
    numTotal = len(column)
    for i in column:
        if (i == 1):
            numOnes += 1
    p1 = (float(numOnes) / numTotal)
    pn1 = 1 - p1
    return (p1, pn1)

# Find the variable and value that give the most information gain
# return threshold value, column number, data > threshold, data < threshold, and best information gain
def findTheta(data):
    xj = 0
    bestInfoGain = 0
    theta = 0
    finalg = []
    finals = []
    for column in range(data.shape[1]):
        if (column != 0):
            currentCol = data.T[column]
            sortedCol = np.sort(currentCol)
            (totalp1, totalpn1) = countOnes(data.T[-1])
            totalHs = entropy(totalp1, totalpn1)
            for testThreshold in sortedCol:
                # Calculate best information gain
                greater = []
                smaller = []
                # for every data array
                for i in data:
                    if i[column] >= testThreshold:
                        greater.append(i)
                    else:
                        smaller.append(i)
                fGreat = float(len(greater)) / len(currentCol)
                fSmall = float(len(smaller)) / len(currentCol)
                greater = np.array(greater)
                smaller = np.array(smaller)
                if(len(greater) == 0):
                    (gp1, gpn1) = (0, 0)
                else:
                    (gp1, gpn1) = countOnes(greater.T[-1])
                # print gp1, gpn1
                gHs = entropy(gp1, gpn1)
                if(len(smaller) == 0):
                    (sp1, spn1) = (0, 0)
                else:
                    (sp1, spn1) = countOnes(smaller.T[-1])
                # print sp1, spn1
                sHs = entropy(sp1, spn1)
                infoGain = (totalHs - (sHs * fSmall) - (gHs * fGreat))
                if (infoGain > bestInfoGain):
                    bestInfoGain = infoGain
                    theta = testThreshold
                    xj = column
                    finalg = np.asarray(greater)
                    finals = np.asarray(smaller)
    if bestInfoGain != 0:
        print "Best info gain: %f | Threshold: %f | var index: %d" % (bestInfoGain, theta, xj)
    return (theta, xj, finalg, finals, bestInfoGain)

# Gets number of errors given threshold value, column number, and data
# Returns number of errors (numberWrong), data > threshold, data < threshold
def getError(theta, column, data):
    greater = []
    smaller = []
    for i in data:
        if (i[column] >= theta):
            greater.append(i)
        else:
            smaller.append(i)
    numberWrong = 0
    for g in greater:
        if (g[-1] == -1):
            numberWrong += 1
    for s in smaller:
        if (s[-1] == 1):
            numberWrong += 1
    return (numberWrong, greater, smaller)

# Returns number wrong on the left side (the number of errors on the side > threshold)
def getLeftError(theta, column, data):
    numberWrong = 0
    numberWright = 0
    greater = []
    for i in data:
        if (i[column] >= theta):
            greater.append(i)
    for g in greater:
        if (g[-1] == -1):
            numberWrong += 1
        else:
            numberWright += 1
    if (numberWright < numberWrong):
        return numberWright
    else:
        return numberWrong

# Returns number wrong on the right side (the number of errors on the side < threshold)
def getRightError(theta, column, data):
    numberWrong = 0
    numberWright = 0
    smaller = []
    for i in data:
        if (i[column] < theta):
            smaller.append(i)
    for g in smaller:
        if (g[-1] == 1):
            numberWrong += 1
        else:
            numberWright += 1
    if (numberWright < numberWrong):
        return numberWright
    else:
        return numberWrong

# Build the binary tree
def doDepth(data, currentDepth, depth):
    rootNode = Node()
    rootNode.depth = currentDepth
    rootNode.data = data
    if (depth > currentDepth):
        (theta, col, g, s, infogain) = findTheta(data)
        rootNode.setVals(theta, col)
        if(infogain == 0):
            return rootNode
        else:
            rootNode.leftError = getLeftError(theta, col, data)
            rootNode.rightError = getRightError(theta, col, data)
            rootNode.left = doDepth(g, (currentDepth+1), depth)
            rootNode.right = doDepth(s, (currentDepth+1), depth)
            return rootNode
    else:
        return rootNode

# Test test-data on the binary tree
def testData(data, root):
    numberWrong = 0
    (theta, col) = root.getVals()
    (wrong, g, s) = getError(theta, col, data)
    if (root.isLeaf()):
        return numberWrong
    elif (root.left.isLeaf() and root.right.isLeaf()):
        return wrong
    elif root.left.isLeaf():
        numberWrong += getLeftError(theta, col, data)
        numberWrong += testData(s, root.right)
    elif root.right.isLeaf():
        numberWrong += getRightError(theta, col, data)
        numberWrong += testData(g, root.left)
    else:
        numberWrong += testData(g, root.left)
        numberWrong += testData(s, root.right)
    return numberWrong

# Traverse the tree for error information on training data, built-in when we built the tree
def getTreeError(root):
    numberWrong = 0
    if root.isLeaf():
        return numberWrong
    elif (root.left.isLeaf() and root.right.isLeaf()):
        numberWrong += root.leftError
        numberWrong += root.rightError
    elif root.left.isLeaf():
        numberWrong += root.leftError
        numberWrong += getTreeError(root.right)
    elif root.right.isLeaf():
        numberWrong += root.rightError
        numberWrong += getTreeError(root.left)
    else:
        numberWrong += getTreeError(root.right)
        numberWrong += getTreeError(root.left)
    return numberWrong



def main():
    train1 = readFile("Subject_1.csv")
    train2 = readFile("Subject_4.csv")
    train3 = readFile("Subject_6.csv")
    train4 = readFile("Subject_9.csv")
    dataTrain = np.concatenate((train1, train2, train3, train4), axis=0)
    print dataTrain.shape
    lt1 = readFile("list_1.csv")
    lt2 = readFile("list_4.csv")
    lt3 = readFile("list_6.csv")
    lt4 = readFile("list_9.csv")
    # trainIndex = np.concatenate((lt1, lt2, lt3, lt4), axis=0)
    # dataTrain = np.hstack((trainIndex.reshape(trainIndex.shape[0], 1), dataTrain))
    positiveData = np.zeros((1, dataTrain.shape[1]*7))
    #grab 7 data points if positive instance
    for i in range(dataTrain.shape[0]):
        if i > 7: #make sure there is 7 data points to grab from
            if dataTrain[i][-1] == 1: # If data is a positive instance
                # print dataTrain[i].reshape(1,10)
                newData = np.zeros((1, dataTrain.shape[1]))
                newData = np.concatenate((newData, dataTrain[i-6].reshape(1,dataTrain.shape[1])), axis=0)
                newData = np.concatenate((newData, dataTrain[i-5].reshape(1,dataTrain.shape[1])), axis=0)
                newData = np.concatenate((newData, dataTrain[i-4].reshape(1,dataTrain.shape[1])), axis=0)
                newData = np.concatenate((newData, dataTrain[i-3].reshape(1,dataTrain.shape[1])), axis=0)
                newData = np.concatenate((newData, dataTrain[i-2].reshape(1,dataTrain.shape[1])), axis=0)
                newData = np.concatenate((newData, dataTrain[i-1].reshape(1,dataTrain.shape[1])), axis=0)
                newData = np.concatenate((newData, dataTrain[i].reshape(1,dataTrain.shape[1])), axis=0)
                newData = np.delete(newData, 0, 0)
                newData = newData.reshape(1,(newData.shape[1])*7)
                positiveData = np.concatenate((positiveData, newData), axis=0)

    print positiveData.shape



    # dataTest = readFile("knn_test.csv")

    # Stump
    #
    # print "Problem 2.1 \nSTUMP: "
    # (thresh, col, g, s, infogain) = findTheta(dataTrain)
    # (placeholder, p, q) = getError(thresh, col, dataTrain)
    # print "Training error rate: %f" % (float(placeholder) / dataTrain.shape[0])
    # # (thresh, col, g, s, infogain) = findTheta(dataTest)
    # # (placeholder, p, q) = getError(thresh, col, dataTest)
    # # print "Testing error rate: %f" % (float(placeholder) / dataTest.shape[0])
    # train = []
    # test = []
    #
    # print "Problem 2.2"
    # for depth in range(1,7):
    #     print "Training data with a depth of %d" % depth
    #     root = doDepth(dataTrain, 0, depth)
    #     numWrong = getTreeError(root)
    #     print "Number Wrong: %d / %d" % (numWrong, dataTrain.shape[0])
    #     rate = float(numWrong) / dataTrain.shape[0]
    #     train.append(rate)
    #
    #     print "Testing data with a depth of %d" % depth
    #     n = testData(dataTest, root)
    #     print "Test Wrong: %d / %d" % (n, dataTest.shape[0])
    #     rate = float(n) / dataTest.shape[0]
    #     test.append(rate)
    #
    # plt.plot(range(1,7), train, 'bx-', label="Training Error")
    # plt.plot(range(1,7), test, 'rx-', label="Testing Error")
    # plt.xlabel('depth')
    # plt.ylabel('Error rate')
    # plt.legend(loc=1, prop={'size':10})
    # plt.savefig("decisionTree.png")
    # print "Graph saved at 'decisionTree.png'"


    return



if __name__ == "__main__":
    main()
