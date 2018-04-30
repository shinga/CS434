# @Author: Arthur Shing
# @Date:   2018-04-29T16:41:18-07:00
# @Filename: decision_tree.py
# @Last modified by:   Arthur Shing
# @Last modified time: 2018-04-29T22:11:50-07:00

import matplotlib
matplotlib.use('Agg')


import numpy as np
import matplotlib.pyplot as plt
import math


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
            # testThreshold = sortedCol[50]
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
                    finalg = greater
                    finals = smaller
                    # Keep testThreshold
                    # Keep column index
    print "Best info gain: %f | Threshold: %f | var index: %d" % (bestInfoGain, theta, xj)
    # (gp1, gpn1) = countOnes(smaller)
    # print greater.shape


    return (theta, xj, finalg, finals)

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

def doDepth(data, depth, thresholdArray, columnArray):

    root = data
    (theta, col, g, s) = findTheta(root)
    if (depth != 0):
        (gtheta, gcol, gg, gs) = doDepth(g, (depth-1), )
        (stheta, scol, sg, ss) = doDepth(s, (depth-1))
        depth -= 1
    else:
        thresholdArray.append(theta)
        columnArray.append(col)
        return (thresholdArray, columnArray)





def main():
    dataTrain = readFile("knn_train.csv")
    dataTest = readFile("knn_test.csv")


    (thresh, col, g, s) = findTheta(dataTrain)
    print getError(thresh, col, dataTrain)
    (thresh, col, g, s) = findTheta(dataTest)
    print getError(thresh, col, dataTest)

    depth = 1
    doDepth(dataTrain, dataTest, depth)


    return



if __name__ == "__main__":
    main()
