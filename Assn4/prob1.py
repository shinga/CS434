# @Author: Arthur Shing
# @Date:   2018-05-22T15:21:11-07:00
# @Filename: prob1.py
# @Last modified by:   Arthur Shing
# @Last modified time: 2018-05-22T18:44:08-07:00
import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from time import sleep
import random


def loadData(fileName):
    print("Loading data...")
    data = np.loadtxt(fileName, delimiter=",")
    print("Loaded.")
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


def main():
    x = loadData("data-1.txt")
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
