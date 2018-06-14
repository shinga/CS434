# @Author: Arthur Shing
# @Date:   2018-05-22T15:21:11-07:00
# @Filename: prob1.py
# @Last modified by:   Arthur Shing
# @Last modified time: 2018-06-13T13:58:03-07:00
import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from time import sleep
import random


def readFile(fileName):
    try:
        data = np.loadtxt(fileName, delimiter=",", usecols=(1,2,3,4,5,6,7,8,9))
    except:
        data = np.loadtxt(fileName, delimiter=",")
    s = data.shape
    print str(s) + " loaded"
    return data

def assignClusters(data, k, c):
    clusterAssignments = []
    clusterNums = range(k)
    for i in clusterNums:
        clusterAssignments.append([])
    for x in data:
        mindist = 999999
        mindistInd = 0
        for j in clusterNums:
            dist = np.linalg.norm(x-c[j])
            if dist < mindist:
                mindist = dist
                mindistInd = j
        clusterAssignments[mindistInd].append(x)
    return clusterAssignments
    # clusterAssignments = np.asarray(clusterAssignments)


def kmeans(data, k):
    c = []
    cIndex = []
    for i in range(k):
        rand = random.randrange(data.shape[0])
        cIndex.append(rand)
        c.append(data[rand])
    c = np.asarray(c)
    print c.shape

    SSElist = []
    prevSSE = 0
    SSE = 1
    ITERATIONS = 10
    for i in range(ITERATIONS):
    # while prevSSE != SSE:
        prevSSE = SSE
        clusterAssignments = assignClusters(data, k, c)
        means = []
        SSE = []
        for i in range(k):
            mean = reassignCentroid(clusterAssignments[i])
            means.append(mean)
            devs = calcDeviation(clusterAssignments[i], mean)
            # print len(devs)
            SSE.append(np.sum(np.square(devs)))
        SSE = sum(SSE)
        print SSE
        c = means
        SSElist.append(SSE)

    return SSElist


def reassignCentroid(clusterAssignments):
    mean = np.mean(clusterAssignments, axis=0)
    return mean
    # print clusterAssignments

def calcDeviation(data, mean):
    dev = []
    for x in data:
        dev.append(np.subtract(x, mean))
    return dev


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
    trainIndex = np.concatenate((lt1, lt2, lt3, lt4), axis=0)
    dataTrain = np.hstack((trainIndex.reshape(trainIndex.shape[0], 1), dataTrain))

    x = normalize(dataTrain)
    print x[0]
    print x[1]
    print x.shape

    k = 2
    sses = kmeans(x, k)
    plotdis(sses)
    plt.savefig("sse_part1.png")
    plt.figure()
    # part 2
    sses = []
    for k in range(2,11):
        sses.append(kmeans(x,k)[-1])

    # plotdis(sses)
    plt.plot(range(2,11), sses)
    plt.xlabel("k")
    plt.ylabel("SSE")
    plt.savefig("sse_part2.png")


def plotdis(sses, k=2):
    plt.plot(range(len(sses)), sses, label=("k = " + str(k)))
    plt.xlabel("Iteration")
    plt.ylabel("SSE")
    plt.legend()
    # for i in x:
    #     showinmage(i)
    #     sleep(3)


def showimage(data):
    plt.figure()
    plt.imshow(np.reshape(data, (28,28)))
    plt.savefig("image.png")
    print("Saving image as image.png...")



if __name__ == '__main__':
    main()
