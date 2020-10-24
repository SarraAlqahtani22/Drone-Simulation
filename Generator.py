# Assumption: Image is square
# For the generator, we are going to use different colors to represent different
# ROI in a object recognition neural network. There should be three types of ROI:
# Real object(true positive), obstacle/reduce(False Negative),
# fake object / enhance(False Positive) and Background(True Negative)

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import transforms as trs
from matplotlib.patches import Ellipse
import matplotlib as mpl
import cv2


# Fake object [color, alpha]
def enhance():
    return ['#ff007f', 0.5]


# Real object [color, alpha]
def real():
    return ['r', 0.5]


# obstacle [color, alpha]
def reduce():
    return ['b', 0.9]


def row_info(picture, index):
    return picture[index, :, 2] - picture[index, :, 0] / 2 - picture[index, :, 1] / 2


# generate the position, size and angle of rotation for each feature
def feature_gen(k=200, n=20, r=15):
    loc = []
    for _ in range(n):
        loc.append([np.random.randint(0, k), np.random.randint(0, k),
                    np.random.randint(1, r), np.random.randint(0, 180)])
    return loc


if __name__ == '__main__':
    # Edit graphic parameters here
    # k1: number of sub-region on length
    k1 = 200

    # [number of feature, maximum size of the feature]
    real_pram = [15, 15]
    obstacle_pram = [45, 30]
    fake_pram = [7, 15]

    # 5.5 = 2000 pixel at dpi = 200. May scale up as demanded
    myDpi = 400
    pic_edge_len = 5.5
    mpl.rcParams['figure.dpi'] = myDpi
    fig = plt.figure(figsize=(pic_edge_len, pic_edge_len), dpi=myDpi, facecolor='w')
    ax = fig.add_axes([0, 0, 1, 1])
    plt.autoscale(False, tight=True)
    plt.xlim(0, k1)
    plt.ylim(0, k1)


    def load_feature(feature, shape='circle', target=None, feature_type='gradient'):

        if target is None:
            target = real()

        for dot in feature:
            if feature_type == 'gradient':
                num_steps = dot[2] * 2
                # We used equal stride for the gradient in this case.
                # It is also possible to use a list of pre-defined alpha value directly
                # Direct superimposing is made when making the gradient effect (e.g given a
                # 8*6 oval centered at (0,0) and then draw another 4*3 oval centered at (0,0))
                for steps in range(1, num_steps + 1):
                    coefficient = steps / num_steps * 2
                    if shape == 'circle':
                        ax.add_artist(
                            plt.Circle((dot[0], dot[1]), dot[2], ec=None, color=target[0],
                                       alpha=target[1] * 2 / num_steps))

                    elif shape == 'oval' or 'ellipse':
                        width_height_ratio = 3 / 4
                        ax.add_artist(
                            Ellipse((dot[0], dot[1]), width=dot[2] * coefficient,
                                    height=dot[2] * width_height_ratio * coefficient, angle=dot[3],
                                    edgecolor=None, facecolor=target[0], alpha=target[1] * 2 / num_steps))

            elif shape == 'circle':
                ax.add_artist(plt.Circle((dot[0], dot[1]), dot[2], ec='none', color=target[0], alpha=target[1]))
            elif shape == 'oval' or 'ellipse':
                width_height_ratio = 3 / 4
                ax.add_artist(
                    Ellipse((dot[0], dot[1]), width=dot[2],
                            height=dot[2] * width_height_ratio, angle=dot[3],
                            edgecolor=None, facecolor=target[0], alpha=target[1]))
            # More custom procedures for adding new features onto the graph can be added here .


    # RawMap: the list for real feature
    RawMap = feature_gen(k1, real_pram[0], real_pram[1])

    # Generating ground truth and save it
    load_feature(RawMap, shape='oval')
    ax.axis('square', adjustabble='box-force')
    ax.plot()
    region = ax.transData.transform([(0, 0), (k1, k1)])
    BBOX = trs.Bbox(region / myDpi)
    fig.savefig("GT_Full", dpi=myDpi, facecolor='w', bbox_inches=BBOX,
                pad_inches=0)

    # Generating all the obstacle and fake target then save it
    Obstacle1 = feature_gen(k1, obstacle_pram[0], obstacle_pram[1])
    fakeTarget = feature_gen(k1, fake_pram[0], fake_pram[1])

    load_feature(Obstacle1, shape='oval', target=reduce())
    load_feature(fakeTarget, shape='oval', target=enhance())
    ax.plot()
    plt.show()
    fig.savefig("wFakeObjects_Full", dpi=myDpi, facecolor='w', bbox_inches=BBOX,
                pad_inches=0)

    # Also generating the information map for ground truth: what we see when closer look is taken
    from joblib import Parallel, delayed

    rawImg = cv2.imread('GT_Full.png')
    GT_InfoMap = np.array(Parallel(n_jobs=4)(delayed(row_info)(rawImg, i)
                                             for i in range(rawImg.shape[1])), dtype=np.uint8)

    np.save('GT_Info_Full', GT_InfoMap)
    cv2.imwrite('GT_Info_Full.png', GT_InfoMap)
