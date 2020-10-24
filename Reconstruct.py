# Visualization of the belief information map.
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np


def from_file(filename='wFakeObjects_Full+s&p+Gaussian.npy'):
    return np.load(filename)


if __name__ == '__main__':
    Map = from_file()
    edgeSize = len(Map)

    boxSize = 10  # same as the one in Interpreter.py
    k = 200  # k is the coordinate resolution(the number of region on each side)
    unitLength = k / edgeSize
    myDpi = 400
    mpl.rcParams['figure.dpi'] = myDpi

    # 5.5 = 2000 pixel at dpi = 200. May scale up as demanded
    fig = plt.figure(figsize=(5.5, 5.5), dpi=myDpi, facecolor='w')
    ax = fig.add_axes([0, 0, 1, 1])
    plt.autoscale(False, tight=True)
    plt.xlim(0, k)
    plt.ylim(0, k)

    counterY = edgeSize

    for i in Map:
        counterY -= 1
        counterX = 0
        for j in i:
            ax.add_artist(
                plt.Rectangle((counterX, counterY * unitLength), unitLength, unitLength, alpha=j / boxSize ** 2))
            counterX += unitLength
    ax.plot()
    plt.show()
