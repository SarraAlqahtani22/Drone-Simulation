# This program will add salt and pepper noise to the image.
import numpy as np
import cv2


# density controls the total strength of noise
# portion controls the proportion sand noise amount all
def salt_noisy(image, density=0.03, portion=0.5):
    s_vs_p = portion
    amount = density
    out = image.copy()

    # Salt mode
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i, int(num_salt))
              for i in image.shape]
    out[tuple(coords)] = 255

    # Pepper mode
    num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i, int(num_pepper))
              for i in image.shape]
    out[tuple(coords)] = 0

    return out


if __name__ == '__main__':
    img0 = cv2.imread('GT_Full.png')
    img1 = salt_noisy(cv2.imread('wFakeObjects_Full.png'))

    cv2.imwrite('wFakeObjects_Full+s&p.png', img1)
