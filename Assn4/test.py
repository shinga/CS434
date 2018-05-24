# @Author: Arthur Shing
# @Date:   2018-05-23T17:31:43-07:00
# @Filename: test.py
# @Last modified by:   Arthur Shing
# @Last modified time: 2018-05-23T17:43:52-07:00
import sys
import numpy as np
import os
import math
from random import randint
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

dim = 784

def dist(p1, p2):#takes in 2 arrays of any length and returns distance
    distance = 0
    for i in range(0,len(p1)):
        distance += float((float(p1[i])-float(p2[i]))**2)
    return math.sqrt(distance)

def readData(filename):
    x = []
    with open(filename) as f:
        for lines in f:
            lines = lines[:-1]#getting rid of \n
            x.append(lines.split(','))
    x = np.array(x)
    return x.astype(np.int)#convert from string to int

def center(x): #computes center of data
    result = np.mean(x)
    return result

def covariance(x,c):
    result2 = np.cov(x.T)
    return result2

def findEigen(covarimatrix):
	values, vectors = np.linalg.eig(covarimatrix)
	index = 0
	ranking = 0
	sortValues = []
	for i in values:
		sortValues.append((i,index))
		index += 1
	sortValues = sorted(sortValues, key=lambda eigen: eigen[0], reverse=True)
	topTen = sortValues[:10]
	tenVectors = np.array([vectors[topTen[0][1]],vectors[topTen[1][1]],vectors[topTen[2][1]],vectors[topTen[3][1]],vectors[topTen[4][1]],vectors[topTen[5][1]],vectors[topTen[6][1]],vectors[topTen[7][1]],vectors[topTen[8][1]],vectors[topTen[9][1]]])
	return topTen,tenVectors

data = readData("data-1.txt")
testMatrix = np.array
center = center(data)
covar = covariance(data,center)
print(center)
print(covar)
eigVals,eigVec = findEigen(covar)
print eigVals
print eigVec
plt.imshow(np.reshape(covar,(784,784)))
plt.savefig("trst.png")
# plt.show()
