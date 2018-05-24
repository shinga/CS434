# @Author: Arthur Shing
# @Date:   2018-05-23T15:53:41-07:00
# @Filename: prob2.py
# @Last modified by:   Arthur Shing
# @Last modified time: 2018-05-23T18:34:05-07:00
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
    # print v
    # norm = np.vectorize(normalize)
    # v = norm(v)

    for i in range(10):
        showimage(v[i], i)
    showimage(mean, 15)

def convertFloat(data):
    return data.real

def normalize(data):
    max = np.amax(data)
    help = np.vectorize(normHelp)
    return help(data, max)


def normHelp(data, max):
    if max != complex(0.0,0.0):
        return data/max
    else:
        return complex(0.0,0.0)




def showimage(data, number):
    plt.figure()
    plt.imshow(np.reshape(data, (28,28)))
    plt.savefig("eig" + str(number) + ".png")
    print("Saving image as eig" + str(number) + ".png...")





if __name__ == '__main__':
    main()
