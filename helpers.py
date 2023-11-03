import os
import math
import cv2 as cv
import numpy as np


def get_mean_angle(img_gray):
    """
    input: grayscale image (cv2 object)
    - looks for lines in images 
    - calculates min of the difference from the two orthogonal lines
    - returns the mean of them over all lines
    """
    img_edges = cv.Canny(img_gray, 100, 100, apertureSize=3)
    lines = cv.HoughLinesP(img_edges, 1, math.pi / 180.0, 100, 
                           minLineLength=150, maxLineGap=5)
    angles = []
    try:
        for [[x1, y1, x2, y2]] in lines:
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            angles.append(min(
                [abs(theta - angle) for theta in [0, 90, -90, 180]]
            ))
    except Exception:
        return 90
    if not angles:
        return 90
    else:
        return np.mean(angles)


def sort_tiltness(path, indices, reverse=False):
    angle = dict()
    red = os.listdir(path)
    for i in indices:
        img_i = cv.imread(
            os.path.join(path, red[i]), cv.IMREAD_GRAYSCALE
        )
        angle[i] = get_mean_angle(img_i)
    return list(sorted(indices, key=lambda x: angle[x], 
                       reverse=reverse))


def is_transparent(img):
    """img: image (cv2 object)
    returns True if image has transparent elements 
    or if image is grayscale, otherwise False
    """
    if len(img.shape) == 2:
        return True
    if len(img[:, :, ][0][0]) < 4:
        return False
    alpha = img[:, :, 3]
    for a in alpha:
        if 0 in a:
            return True
    return False


def is_white_bg(img):
    """img: image (cv2 object)
    returns True if image has either 
    - left and right vertical white borders, or
    - top and bottom horizontal white borders
    """
    if len(img.shape) == 2:
        return False
    # check row
    val = 255
    for arr in img[0]:
        for v in arr[:3]:
            val = min(v, val)
    val_horizontal_up = val > 230
    val = 255
    for arr in img[-1]:
        for v in arr[:3]:
            val = min(v, val)
    val_horizontal_down = val > 230
    val_left, val_right = 255, 255
    for i in range(len(img)):
        arr = img[i][0]
        for v in arr[:3]:
            val_left = min(v, val_left)
        arr = img[i][-1]
        for v in arr[:3]:
            val_right = min(v, val_right)
    return (val_horizontal_up and val_horizontal_down) \
        or (val_right > 230 and val_left > 230)


def get_front_area(image):
    """image: grayscale image (cv2 object)
    returns area of the bounding rectangle
    """
    # assuming image is grayscale
    blur = cv.GaussianBlur(image, (3, 3), 0)
    thresh = cv.threshold(blur, 0, 255, 
                          cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]
    x, y, w, h = cv.boundingRect(thresh)
    # cv.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
    return h * w
