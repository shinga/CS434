# @Author: Arthur Shing
# @Date:   2018-04-25T17:07:51-07:00
# @Filename: model_select.py
# @Last modified by:   Arthur Shing
# @Last modified time: 2018-04-25T17:46:04-07:00

import numpy as np
import math
np.set_printoptions(threshold=np.nan)


def readFile(fileName):
    data = np.loadtxt(fileName, delimiter=",")
    s = data.shape
    print s
    return data

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

    return newData




def main():
    data = readFile("knn_train.csv")
    print normalize(data)



    return






if __name__ == "__main__":
    main()
