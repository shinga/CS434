# @Author: Arthur Shing
# @Date:   2018-05-22T15:21:11-07:00
# @Filename: prob1.py
# @Last modified by:   Arthur Shing
# @Last modified time: 2018-05-22T18:13:54-07:00
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


    prevSSE = 0
    SSE = 1
    while prevSSE != SSE:
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
    k = 10
    x = loadData("data-1.txt")
    print x.shape
    kmeans(x, k)
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
