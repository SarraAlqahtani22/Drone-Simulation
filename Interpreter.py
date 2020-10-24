import Simulator as Sim
import numpy as np
from joblib import Parallel, delayed
from scipy.ndimage import gaussian_filter


# If the parameter for parallel processing is not correctly met,
# we will use only 1 processor
def check_parallel(ImgSize, numJobs, boxSize0):
    if ImgSize % (numJobs * boxSize0) == 0:
        return True
    return False


# cv2 uses GBR instead of RGB. and for any color, we let R be the "ROI" degree of certainty
# and GB be the degree of uncertainty. More certainty means more information of interest
# according to the NN. The information here, however, is not normalized and will be normalized later.

# The formula for calculating information is: (R-(B+G)/2)/255, and the information cannot be negative
def calculate_info(index, GT_Img0, n_worker0, boxSize0):
    base = index * int(GT_Img0.shape[0] / n_worker0)
    end = (index + 1) * int(GT_Img0.shape[0] / n_worker0)

    chunkRow = int(GT_Img0.shape[0] / boxSize0 / n_worker0)
    chunkCol = int(GT_Img0.shape[1] / boxSize0)
    GT_InfoMap0 = np.zeros((chunkRow, chunkCol), dtype=float)

    for i in range(base, end):
        for j in range(GT_Img0.shape[1]):
            loc = GT_Img0[i, j]
            val = (loc[2] - (float(loc[0]) + loc[1]) / 2) / 255

            if val > 0:
                GT_InfoMap0[int((i - base) / boxSize0), int(j / boxSize0)] += val

    return GT_InfoMap0


def p_processing_img(GT_Img, BoxSize):
    # Parallel implementation of calculate_info
    n_worker = 4
    results = None
    if check_parallel(GT_Img.shape[0], n_worker, BoxSize):
        results = Parallel(n_jobs=n_worker)(
            delayed(calculate_info)(ind, GT_Img, n_worker, BoxSize) for ind in range(n_worker))
    else:  # not tested
        results = Parallel(n_jobs=1)(
            delayed(calculate_info)(ind, GT_Img, n_worker, BoxSize) for ind in range(n_worker))

    GT_InfoMap = results[0]
    for n in range(1, n_worker):
        GT_InfoMap = np.concatenate((GT_InfoMap, results[n]), axis=0)

    return GT_InfoMap


if __name__ == '__main__':
    # Run after blurring.py
    # reduce the resolution and setting gaussian blurry level
    rows = 200
    cols = 200
    boxSize = 10
    gaussian_sigma = 1

    # Reduce resolution with ICRSsimulator
    Obj_real = Sim.ICRSsimulator('GT_Full.png')
    Obj_blurred = Sim.ICRSsimulator('wFakeObjects_Full+s&p.png')

    if not Obj_blurred.loadImage() or not Obj_real.loadImage():
        print("Error: could not load image")
        exit(0)

    Obj_real.setMapSize(rows, cols)
    Obj_real.createMap()
    Obj_blurred.setMapSize(rows, cols)
    Obj_blurred.createMap()

    Img_real = Obj_real.img
    Img_blurred = Obj_blurred.img

    InfoMap_real = p_processing_img(Img_real, boxSize)
    InfoMap_blurred = p_processing_img(Img_blurred, boxSize)

    # Apply Gaussian blurry
    InfoMap2 = gaussian_filter(InfoMap_blurred, sigma=gaussian_sigma)

    # Save the reduced resolution map with and without being blurred.
    np.save('GT_Info_Convoluted', np.asarray(InfoMap_real), allow_pickle=False)
    np.save('wFakeObjects_Info_Convoluted+s&p+Gaussian', np.asarray(InfoMap2), allow_pickle=False)
