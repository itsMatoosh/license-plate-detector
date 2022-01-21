import math

import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters

"""
    In this file, you need to define plate_detection function.
    To do:
        1. Localize the plates and crop the plates
        2. Adjust the cropped plate images
    Inputs:(One)
        1. image: captured frame in CaptureFrame_Process.CaptureFrame_Process function
        type: Numpy array (imread by OpenCV package)
    Outputs:(One)
        1. plate_imgs: cropped and adjusted plate images
        type: list, each element in 'plate_imgs' is the cropped image(Numpy array)
    Hints:
        1. You may need to define other functions, such as crop and adjust function
        2. You may need to define two ways for localizing plates(yellow or other colors)
"""


def morphology_close(image: np.ndarray):
    """Performs the morphological close operation on an image."""
    structuring_element = np.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ], dtype=np.uint8)
    img_dilated = cv2.dilate(image, structuring_element)
    return cv2.erode(img_dilated, structuring_element)


def plot_lines_on_image(image, lines, x_max, y_max):
    """Plots lines on top of an image."""
    # # Plot the original image
    # Using subplots to be able to plot the lines above it
    fig, ax = plt.subplots()
    plt.figure(figsize=(20, 10))
    ax.imshow(image)

    x_axis = np.linspace(0, x_max, num=x_max + 1)
    for line in lines:
        r = line[0, 0]
        t = line[0, 1]
        y_axis = np.minimum(np.maximum(-(np.cos(t) / np.sin(t)) * x_axis + (r / np.sin(t)), 0), y_max)
        ax.plot(x_axis, y_axis, linewidth=1)

    # Showing the final plot with all the lines
    plt.show()


def find_license_contours(image: np.ndarray):
    """Finds contours of a license plate in an image."""
    # convert img to grey
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # treshold image
    treshold = 50
    ret, img_tresh = cv2.threshold(img_gray, treshold, 255, cv2.THRESH_BINARY)

    # morphological closing
    img_processed = morphology_close(img_tresh)

    # find all contours
    contours, hierarchy = cv2.findContours(img_processed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # filter contour shape
    contours_filtered = []
    for cnt in contours:
        rect = cv2.boundingRect(cnt)
        width = rect[2]
        height = rect[3]
        area = width * height
        aspect_ratio = width/height
        if aspect_ratio > 2.4 and area > 1000:
            contours_filtered.append(cnt)

    return contours_filtered


def crop_to_bound(image, bound):
    """Crops image to a bounding box."""
    bnd_x, bnd_y, bnd_w, bnd_h = bound[0], bound[1], bound[2], bound[3]
    return image[bnd_y:bnd_y + bnd_h, bnd_x:bnd_x + bnd_w]


def crop_license_plate(image_bgr: np.ndarray, contour_image: np.ndarray, contour):
    """Crops a license plate image for recognition."""
    # get plate bounding box
    bound = cv2.boundingRect(contour)
    bnd_w, bnd_h = bound[2], bound[3]

    # check if anything is left
    if bnd_w == 0 or bnd_h == 0:
        return None

    # crop
    img_cropped = crop_to_bound(image_bgr, bound)
    cnt_cropped = crop_to_bound(contour_image, bound)

    # rho constraints
    rho_max = np.hypot(bnd_w, bnd_h)

    # detect horizontal lines
    r_dim = 100
    theta_dim = 50
    theta_max = 9/16 * math.pi
    theta_min = 7/16 * math.pi
    threshold = 28

    # find lines
    lines_horizontal = cv2.HoughLines(cnt_cropped, rho_max/r_dim, theta_max/theta_dim, threshold,
                           min_theta=theta_min, max_theta=theta_max)

    # check if any lines found
    if lines_horizontal is None or len(lines_horizontal) == 0:
        return None

    average_t = np.average(lines_horizontal[:, :, 1])
    t_change = np.pi/2 - average_t

    # rotate cropped image
    rotated = imutils.rotate(img_cropped, np.degrees(-t_change))

    # crop again to minimize space around
    rotated_pp = preprocess_image(rotated)
    contours = find_license_contours(rotated_pp)
    if len(contours) == 0:
        return None
    bound_final = cv2.boundingRect(contours[0])
    if bound_final[2] == 0 or bound_final[3] == 0:
        return None
    rotated = crop_to_bound(rotated, bound_final)
    return rotated


def preprocess_image(image: np.ndarray):
    """Preprocesses the input image"""
    # Convert to HSI/HSV
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define color range
    color_min = np.array([10, 100, 90])
    color_max = np.array([40, 255, 255])

    # Segment only the selected color from the image and leave out all the rest (apply a mask)
    mask = cv2.inRange(image_hsv, color_min, color_max)

    # Plot the masked image (where only the selected color is visible)
    image_masked = cv2.bitwise_and(image_hsv, image_hsv, mask=mask)

    return cv2.cvtColor(image_masked, cv2.COLOR_HSV2RGB)


def contour_image(contours, id, width, height):
    """Creates an image with the given contours drawn on it."""
    img_cnt = np.zeros((height, width), dtype='uint8')
    return cv2.drawContours(img_cnt, contours, id, 255, 1)


def plate_detection(image: np.ndarray):
    """Performs localization on the given image"""
    # preprocess image
    image_processed = preprocess_image(image)

    # find contours
    contours = find_license_contours(image_processed)

    plate_imgs = []
    for i in range(len(contours)):
        # find lines in contours to get rotation
        img_cnt = contour_image(contours, i, image.shape[1], image.shape[0])

        # crop license plate
        plate = crop_license_plate(image, img_cnt, contours[i])
        if plate is not None:
            plate_imgs.append(plate)

    return plate_imgs