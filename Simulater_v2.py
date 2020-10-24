import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

# setting the image size here
edge_len = 2000
n_processor = 4


# helper method
def goi_kernel(i, h_start, h_end, center, half_width, half_height, direction_angle):
    index = []
    for j in range(h_start, h_end):
        # Oval formula with rotation angle integration

        distance = ((i - center[0]) * np.cos(direction_angle) + (j - center[1]) * np.sin(
            direction_angle)) ** 2 / half_width ** 2 + (
                           (i - center[0]) * np.sin(direction_angle) -
                           (j - center[1]) * np.cos(direction_angle)) ** 2 / half_height ** 2

        if distance <= 1:
            index.append([i, j])

    return np.array(index, dtype='int16')


# helper method
def get_oval_index(center, direction_angle, half_height, half_width):
    center = np.array(center, dtype='int16')
    index = []
    buffer = max(half_width, half_height)
    w_start, w_end = center[0] - buffer, center[0] + buffer + 1
    h_start, h_end = center[1] - buffer, center[1] + buffer + 1
    half_width, half_height = int(half_width), int(half_height)

    distance_j = Parallel(n_jobs=n_processor)(
        delayed(goi_kernel)(i, h_start, h_end, center, half_width, half_height, direction_angle)
        for i in range(w_start, w_end))

    for i in range(1, w_end - w_start):
        if len(distance_j[i].shape) == 2:
            index.append(distance_j[i])

    index = np.concatenate(index, axis=0)

    return index


def plot_oval(canvas_temp, center, direction_angle, half_height, half_width,
              peak, margin, oval_id, num_level, manual=False):
    """

    :param canvas_temp: the 3D array containing ROI id and probabilities
    :param center: (row,column) of ellipse center
    :param direction_angle: the rotation angle of the ellipse
    :param half_height: half of the span fo the ellipse on y-axis
    :param half_width: half of the span of the ellipse on x-axis
    :param peak: probability at center
    :param margin: probability at margin
    :param oval_id: id of the ROI
    :param num_level: the total number of gradient levels for the ellipse
    :param manual: the ellipse information is added manually rather than
                    a random number generator
    :return canvas_temp: the updated canvas 3D array
    """
    # layer 0 is id
    # layer 1 is information
    if not 0 <= margin < peak <= 1:
        print("Please enter parameter margin and peak  properly")
    else:

        gradient_stride = np.ceil(min(half_width, half_height) / num_level)
        delta = (peak - margin) / (np.floor(min(half_width, half_height) / gradient_stride))
        index = get_oval_index(center, direction_angle, half_height, half_width)

        if len(index[canvas_temp[index[:, 0], index[:, 1], 0] != 0]) == 0:
            canvas_temp[index[:, 0], index[:, 1], 0] = oval_id

            while min(half_width, half_height) > gradient_stride:
                ratio = half_height / half_width
                half_width -= gradient_stride
                half_height -= int(ratio * gradient_stride)
                index = get_oval_index(center, direction_angle, half_height, half_width)
                canvas_temp[index[:, 0], index[:, 1], 1] += delta
        else:
            if manual:
                print("Overlapping occurred")

    return canvas_temp


def visualization(canvas_temp, top_color=('#ffffff', '#ff0000', '#00ff00')):
    """
    :param canvas_temp: the 3-D canvas array containing id and probability
    :param top_color: tuple form, RGB color in hex, the color at ROI center

    :return img: a png-formatted image with RGB channels
    """
    if canvas_temp.shape[2] != 2:
        print('visualization failed, please make sure canvas_temp is of shape ( , ,2)')
        return False
    max_id = np.amax(canvas_temp[:, :, 0])
    if max_id >= len(top_color):
        print('Need a longer color tuple')
        return False
    img = np.ones((edge_len, edge_len, 3), dtype='uint8') * 255

    for i in range(1, int(max_id) + 1):
        h = top_color[i].lstrip('#')
        rgb = tuple(int(h[j:j + 2], 16) for j in (0, 2, 4))
        loc = np.argwhere(canvas_temp[:, :, 0] == i)
        for j in range(3):
            img[loc[:, 0], loc[:, 1], j] = \
                rgb[j] * canvas_temp[loc[:, 0], loc[:, 1], 1] + \
                (1 - canvas_temp[loc[:, 0], loc[:, 1], 1]) * 255
    return img


if __name__ == "__main__":
    # layer 0  is coloring, layer 1 is ID, and layer 2 is information
    canvas = np.zeros((edge_len, edge_len, 2), dtype='float32')
    # plot_oval(canvas,
    #           index,
    #           rotation_angle,
    #           width_length,
    #           height_length,
    #           center probability,margin_probability,
    #           OvalID,
    #           numOfGradient)
    # You may use a for loop to randomly generate feature
    canvas = plot_oval(canvas, [700, 800], np.pi / 4, 100, 60, 1, 0.4, 1, 8, manual=True)
    canvas = plot_oval(canvas, [1400, 350], np.pi / 8, 120, 100, 1, 0.6, 2, 10, manual=True)
    np.save('map_version2.npy', canvas)
    plt.imshow(canvas[:, :, 1] * 255, cmap='gray')
    plt.show()
    plt.imshow(canvas[:, :, 0])
    plt.show()

    img_output = visualization(canvas)
    plt.imshow(img_output)
    plt.show()
