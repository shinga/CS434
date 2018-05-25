# @Author: Arthur Shing
# @Date:   2018-05-23T15:53:41-07:00
# @Filename: prob2.py
# @Last modified by:   Arthur Shing
# @Last modified time: 2018-05-24T21:01:10-07:00
import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from time import sleep
import random
np.set_printoptions(threshold=np.nan)


def loadData(fileName):
    print("Loading data...")
    data = np.loadtxt(fileName, delimiter=",")
    print("Loaded.")
    return data




def main():
    x = loadData("data-1.txt")
    print x.shape

    mean = np.mean(x, axis=0)

    # print mean.shape
    # xa = x - mean
    cov = np.cov(x.T)
    # cov = np.cov(xa.T)
    # print cov


    # w = eigenvalues
    # v = eigenvectors
    w, v = np.linalg.eigh(cov)
    index = w.argsort()[::-1]
    # print index
    w = w[index] #sort
    v = v[:,index]
    print("Top 10 Eigen Values: ")
    print w[:10]
    # print v
    v = v.T
    # norm = np.vectorize(normalize)
    # v = norm(v)
    # v = np.apply_along_axis(normalize, 1, v)
    # print v[:10]
    # print v[:10].shape


    for i in range(10):
        showimage(v[i], i)
    showimage(mean, 15)


    b = []
    for i in v[:10]:
        b.append(np.dot(x, i).argsort()[::-1])
    ind = 100
    for vector in b:
        print vector[0]
        # sorted_x = x[vector]
        showimage(x[vector[0]], ind)
        ind += 1
    # for vector in b:
    #     print vector[0]
    #     sorted_x = x[vector]
    #     showimage(sorted_x[0], ind)
    #     ind += 1
    # sorted_x = x[b[0]]

    # print sorted_x[0]
    # PRINT

def convertFloat(data):
    return data.real

def normalize(data):
    # print data
    max = np.amax(data)
    min = np.amin(data)
    help = np.vectorize(normHelp)
    return help(data, max, min)


def normHelp(data, max, min):
    if (float(max) - float(min)) != 0:
        # print data, max, min
        return (float(data) - float(min)) / (float(max) - float(min))
    else: return 0




def showimage(data, number):
    plt.figure()
    plt.imshow(np.reshape(data, (28,28)))
    plt.savefig("eig" + str(number) + ".png")
    print("Saving image as eig" + str(number) + ".png...")
    plt.close()





if __name__ == '__main__':
    main()
