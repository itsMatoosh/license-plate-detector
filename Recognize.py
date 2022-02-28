import os

import cv2
import numpy as np

from tools import gradient

# cached sift database of character images
sift_database = {}

def sift_descriptor(image):
    result = np.zeros(128)
    # Take only 16x16 window of the picture from the center

    # get gradient magnitudes and directions
    grad_mag, grad_dir = gradient.get_gradient(image)

    # Iterate over every pixel
    for y in range(16):
        for x in range(16):
            # Add the direction of the edge to the feature vector, scaled by its magnitude

            # get cell coords
            x_cell = x // 4
            y_cell = y // 4

            # get direction bin
            p_dir = grad_dir[y, x]
            dir_bin = int(p_dir / (2 * np.pi) * 8)

            # get pixel magnitude
            p_mag = grad_mag[y, x]

            # add to result
            result[y_cell * 4 * 8 + x_cell * 8 + dir_bin] += p_mag

    return result


def compare_euclidean_norm(a, b):
    return np.linalg.norm(a - b)


def NN_SIFT_classifier(image, database):
    # target dimensions
    target_h = 85
    target_w = 100

    # image data
    height = image.shape[0]
    width = image.shape[1]
    aspect_ratio = width / height

    # find dimensions to resize
    if aspect_ratio > 1:
        # dash image
        # resize to be width 50
        new_w = target_w // 2
        scale = new_w / width
        new_h = int(height * scale)
    else:
        # digit image
        # resize to be height 85
        new_h = target_h
        scale = new_h / height
        new_w = int(width * scale)

    # scale image
    image = cv2.resize(image, (new_w, new_h))

    # adjust to be the same size as database picture
    canvas = np.zeros((target_h, target_w))
    if new_w > target_w:
        new_w = target_w
    if new_h > target_h:
        new_w = target_h

    start_y = (target_h - new_h) // 2
    canvas[start_y:start_y + new_h, 0:new_w] = image[:, 0:new_w]

    # nearest neighbor classifier
    distance = {}

    # resize the image if needed to fit the training set
    image = cv2.resize(canvas, (16, 16))

    # find the SIFT descriptor of the test image
    sift = sift_descriptor(image)

    # measure the similarity between the test image descriptor & all the database images' descriptors
    for key in database:
        distance[key] = compare_euclidean_norm(sift, database[key])

    # Steps: 6- Sort the labels based on the similarity
    distance = dict(sorted(distance.items(), key=lambda x: x[1]))

    # Steps: 7- print the label & the distances of the sorted list
    # print("new char")
    # for key in distance:
    #     print(key + ": " + str(distance[key]))

    # Return the label of the classification with the minimum distance & the disatance 
    return min(distance, key=distance.get), distance


def isodata_thresholding(image, epsilon=2):
    # Compute the histogram and set up variables
    hist = cv2.calcHist([image], [0], None, [256], [0, 256]).reshape(256)
    tau = np.random.randint(hist.nonzero()[0][0], 256 - hist[::-1].nonzero()[0][0])
    old_tau = -2 * epsilon

    # Iterations of the isodata thresholding algorithm
    while abs(tau - old_tau) >= epsilon:
        # Calculate m1
        m1 = 0
        if tau > 1:
            for i in range(tau):
                m1 += i * hist[i]
            m1 = m1 / hist[:tau].sum()
        # Calculate m2
        m2 = 0
        if tau < 255:
            for i in range(tau, 256):
                m2 += i * hist[i]
            m2 = m2 / hist[tau:].sum()

        # Calculate new tau
        old_tau = tau
        tau = int((m1 + m2) / 2)

    # Threshold the image based on last tau
    ret, foreground = cv2.threshold(image, tau, 255, cv2.THRESH_BINARY)
    background = 255 - foreground
    return background


def segment_characters(tresh_image):
    """Segments a tresholded image into separate characters."""
    # find all contours
    contours, hierarchy = cv2.findContours(tresh_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # filter contour shape
    area_img = tresh_image.shape[0] * tresh_image.shape[1]
    extracted = {}
    for cnt in contours:
        # get bounding rect of the contour
        rect = cv2.boundingRect(cnt)
        bnd_x, bnd_y, bnd_w, bnd_h = rect[0], rect[1], rect[2], rect[3]

        # get area and aspect ratio of rect
        area_rel = bnd_w * bnd_h / area_img

        aspect_ratio = bnd_w / bnd_h
        char = tresh_image[bnd_y:bnd_y + bnd_h, bnd_x:bnd_x + bnd_w]
        if (0.4 < aspect_ratio < 0.7 and area_rel > 0.04) \
                or (1.3 < aspect_ratio < 2.2 and 0.0025 < area_rel < 0.0065):
            # crop character
            extracted[bnd_x] = char

    # sort characters
    chars_sorted = dict(sorted(extracted.items(), key=lambda x: x[0]))
    return chars_sorted.values()


def create_sift_database():
    if len(sift_database) == 0:
        for fname in os.listdir("data/SameSizeLetters"):
            if fname.endswith('.png') or fname.endswith('.jpg') or fname.endswith('.bmp'):
                image = cv2.imread("data/SameSizeLetters/" + fname)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = cv2.resize(image, (16, 16))
                sift = sift_descriptor(image)
                sift_database[fname] = sift

        for fname in os.listdir("data/SameSizeNumbers"):
            if fname.endswith('.png') or fname.endswith('.jpg') or fname.endswith('.bmp'):
                image = cv2.imread("data/SameSizeNumbers/" + fname)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = cv2.resize(image, (16, 16))
                sift = sift_descriptor(image)
                sift_database[fname] = sift
    return sift_database


def segment_and_recognize(plate_imgs):
    # get sift database
    database = create_sift_database()

    # result databse
    res = []

    # Segment image and run NN_Sift_Classifier on each character
    for image in plate_imgs:
        # treshold image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = isodata_thresholding(gray)

        # segment characters
        char_imgs = segment_characters(thresh)

        # match character images to symbols
        matches = []
        letter = None
        for char in char_imgs:
            # use sift to match image
            match, distance = NN_SIFT_classifier(char, database)
            if match[0] == '-':
                letter = None
                matches.append('-')
                continue
            if letter is None:
                if match[0] in ('1', '2', '3', '4', '5', '6', '7', '8', '9', '0'):
                    letter = False
                else:
                    letter = True

                matches.append(match[0])
                continue
            if letter:
                while match[0] in ('1', '2', '3', '4', '5', '6', '7', '8', '9', '0'):
                    distance.pop(match)
                    match = min(distance, key=distance.get)
            else:
                while match[0] not in ('1', '2', '3', '4', '5', '6', '7', '8', '9', '0'):
                    distance.pop(match)
                    match = min(distance, key=distance.get)
            matches.append(match[0])

        res.append(''.join([char for char in matches]))

    return res
