# @Author: Arthur Shing
# @Date:   2018-04-29T16:41:18-07:00
# @Filename: decision_tree.py
# @Last modified by:   Arthur Shing
# @Last modified time: 2018-04-30T19:11:20-07:00

import matplotlib
matplotlib.use('Agg')


import numpy as np
import matplotlib.pyplot as plt
import math

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
    data = np.loadtxt(fileName, delimiter=",")
    s = data.shape
    print str(s) + " loaded"
    return data

def entropy(p1, pn1):
    if(p1 == 0 or pn1 == 0):
        return 0
    l2 = np.log2(p1)
    nl2 = np.log2(pn1)
    Hs = -((p1*l2) + (pn1*nl2))
    # print Hs
    return Hs



def countOnes(column):
    numOnes = 0
    numTotal = len(column)
    for i in column:
        if (i == 1):
            numOnes += 1
    p1 = (float(numOnes) / numTotal)
    pn1 = 1 - p1
    return (p1, pn1)


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
            (totalp1, totalpn1) = countOnes(data.T[0])
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
                    (gp1, gpn1) = countOnes(greater.T[0])
                # print gp1, gpn1
                gHs = entropy(gp1, gpn1)
                if(len(smaller) == 0):
                    (sp1, spn1) = (0, 0)
                else:
                    (sp1, spn1) = countOnes(smaller.T[0])
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
        if (g[0] == -1):
            numberWrong += 1
    for s in smaller:
        if (s[0] == 1):
            numberWrong += 1
    return numberWrong

def getLeftError(theta, column, data):
    numberWrong = 0
    numberWright = 0
    greater = []
    for i in data:
        if (i[column] >= theta):
            greater.append(i)
    for g in greater:
        if (g[0] == -1):
            numberWrong += 1
        else:
            numberWright += 1
    if (numberWright < numberWrong):
        return numberWright
    else:
        return numberWrong

def getRightError(theta, column, data):
    numberWrong = 0
    numberWright = 0
    smaller = []
    for i in data:
        if (i[column] < theta):
            smaller.append(i)
    for g in smaller:
        if (g[0] == 1):
            numberWrong += 1
        else:
            numberWright += 1
    if (numberWright < numberWrong):
        return numberWright
    else:
        return numberWrong

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
    dataTrain = readFile("knn_train.csv")
    dataTest = readFile("knn_test.csv")

    # Stump
    # (thresh, col, g, s, infogain) = findTheta(dataTrain)
    # print getError(thresh, col, dataTrain)
    # (thresh, col, g, s, infogain) = findTheta(dataTest)
    # print getError(thresh, col, dataTest)
    train = []
    test = []
    for depth in range(1,7):
        print "Training data with a depth of %d" % depth
        root = doDepth(dataTrain, 0, depth)
        numWrong = getTreeError(root)
        print "Number Wrong: %d / %d" % (numWrong, dataTrain.shape[0])
        rate = float(numWrong) / dataTrain.shape[0]
        train.append(rate)

        print "Testing data with a depth of %d" % depth
        root = doDepth(dataTest, 0, depth)
        numWrong = getTreeError(root)
        print "Number Wrong: %d / %d" % (numWrong, dataTest.shape[0])
        rate = float(numWrong) / dataTest.shape[0]
        test.append(rate)

    plt.plot(range(1,7), train, 'bx-', label="Training Error")
    plt.plot(range(1,7), test, 'rx-', label="Testing Error")
    plt.xlabel('depth')
    plt.ylabel('Error rate')
    plt.legend(loc=1, prop={'size':10})
    plt.savefig("decisionTree.png")
    print "Graph saved at 'decisionTree.png'"



    # tnode = test.right.left.right.left
    # print tnode.data.T[0]
    # print getTreeError(tnode)
    # tnode.printNode()
    # t = 0
    # for i in tnode.data.T[0]:
    #     if i == 1:
    #         t += 1
    # print t
    # print getTreeError(test)

    return



if __name__ == "__main__":
    main()
