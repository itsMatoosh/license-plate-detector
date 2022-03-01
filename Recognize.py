import math
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

from tools import gradient

number_chars = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']

"""Cached sift database of character images"""
sift_database = np.array([])
sift_database_labels = []

"""Chains of recognized plate data.
    Used for majority voting on final plate output."""
recognize_chains = []
"""Information about which frame a chain started on."""
chain_metadata = []


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


def nn_sift_classifier(image, sift_db, sift_labels, dashes):
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

    # resize the image if needed to fit the training set
    image = cv2.resize(canvas, (16, 16))

    # find the SIFT descriptor of the test image
    sift = sift_descriptor(image)

    # measure the similarity between the test image descriptor & all the database images' descriptors
    distances = np.linalg.norm(sift[None, :] - sift_db, axis=1)

    # get min distance
    min_dist_index = np.argmin(distances)
    min_label = sift_labels[min_dist_index]

    return min_label, distances[min_dist_index]

    # sort the labels based on the similarity
    # distance = dict(sorted(distance.items(), key=lambda x: x[1]))
    # if dashes and "-" in distance:
    #     distance.pop("-")
    # # Return the label of the classification with the minimum distance & the distance
    # return min(distance, key=distance.get), distance


def isodata_thresholding(image, epsilon=2):
    """Perform ISODATA tresholding on a given image."""
    # Compute the histogram and set up variables
    hist = cv2.calcHist([image], [0], None, [256], [0, 256]).reshape(256)
    tau = np.random.randint(hist.nonzero()[0][0], 256 - hist[::-1].nonzero()[0][0])
    old_tau = -2 * epsilon

    # Iterations of the isodata thresholding algorithm
    while abs(tau - old_tau) >= epsilon:
        # Calculate m1
        m1 = 0
        hist_sum = hist[:tau].sum()
        if hist_sum > 0:
            for i in range(tau):
                m1 += i * hist[i]
            m1 = m1 / hist_sum
        # Calculate m2
        m2 = 0
        hist_sum = hist[tau:].sum()
        if hist_sum > 0:
            for i in range(tau, 256):
                m2 += i * hist[i]
            m2 = m2 / hist_sum

        # Calculate new tau
        old_tau = tau
        tau = int((m1 + m2) / 2)

    # Threshold the image based on last tau
    ret, foreground = cv2.threshold(image, tau, 255, cv2.THRESH_BINARY)
    background = 255 - foreground
    return background


def filter_contours(image, contours, min_aspect, max_aspect, min_area, max_area, clear=False):
    img_out = np.copy(image)
    area_img = image.shape[0] * image.shape[1]
    extracted = {}
    x_start = image.shape[1]
    x_end = 0
    for cnt in contours:
        # get bounding rect of the contour
        rect = cv2.boundingRect(cnt)
        bnd_x, bnd_y, bnd_w, bnd_h = rect[0], rect[1], rect[2], rect[3]

        # get area and aspect ratio of rect
        area_rel = bnd_w * bnd_h / area_img

        aspect_ratio = bnd_w / bnd_h
        char = image[bnd_y:bnd_y + bnd_h, bnd_x:bnd_x + bnd_w]
        # filter just letters
        if min_aspect < aspect_ratio < max_aspect \
                and min_area < area_rel < max_area:
            # crop character
            extracted[bnd_x] = char

            # adjust x_start and x_end
            if bnd_x < x_start:
                x_start = bnd_x
            if bnd_x + bnd_w > x_end:
                x_end = bnd_x + bnd_w

            # clear image around the contour
            if clear:
                img_out[:, bnd_x:bnd_x + bnd_w] = 0

    return extracted, img_out, x_start, x_end


def segment_characters(tresh_image):
    """Segments a tresholded image into separate characters."""
    # find all contours
    contours1, hierarchy = cv2.findContours(tresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # extract letters based on contours
    # clear image where extracted
    extracted, tresh_img_cleared, letters_start, letters_end = filter_contours(
        tresh_image, contours1, 0.5, 0.8, 0.04, math.inf, clear=True)

    # clear parts of the img where the dashes wont be
    clear_h = int(tresh_image.shape[0] * 0.3)
    tresh_img_cleared[0:clear_h, :] = 0
    tresh_img_cleared[tresh_img_cleared.shape[0] - clear_h:tresh_img_cleared.shape[0], :] = 0
    tresh_img_cleared[:, 0:letters_start] = 0
    tresh_img_cleared[:, letters_end:-1] = 0

    # plt.imshow(tresh_img_cleared)
    # plt.show()

    # extract dashes based on contours
    contours2, hierarchy = cv2.findContours(tresh_img_cleared, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    extracted_dashes, _, _, _ = filter_contours(tresh_img_cleared, contours2, 1.1, 2.6, 0.003, 0.013)

    sortedChars = dict(sorted(extracted.items(), key=lambda x: x[0]))
    dashIndexes = []
    for d in sorted(list(extracted_dashes.keys())):
        i = 0
        while d > list(sortedChars.keys())[i]:
            i += 1
        dashIndexes.append(i + len(dashIndexes))
    # sort characters
    return sortedChars.values(), dashIndexes


def create_sift_database():
    """Create a new SIFT characters database or reuse an existing one."""
    db = []
    labels = []
    for fname in os.listdir("data/SameSizeLetters"):
        if fname.endswith('.png') or fname.endswith('.jpg') or fname.endswith('.bmp'):
            image = cv2.imread("data/SameSizeLetters/" + fname)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (16, 16))
            sift = sift_descriptor(image)
            db.append(sift)
            labels.append(fname)
    for fname in os.listdir("data/SameSizeNumbers"):
        if fname.endswith('.png') or fname.endswith('.jpg') or fname.endswith('.bmp'):
            image = cv2.imread("data/SameSizeNumbers/" + fname)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (16, 16))
            sift = sift_descriptor(image)
            db.append(sift)
            labels.append(fname)
    return np.array(db), labels


def morph_open(image, open):
    """Applies a morphological opening operation on an image"""
    kernel = np.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]
    ], dtype='uint8')
    return cv2.morphologyEx(image, cv2.MORPH_OPEN if open else cv2.MORPH_CLOSE, kernel)


def segment_and_recognize(plate_imgs):
    # get sift database
    global sift_database, sift_database_labels
    if len(sift_database) == 0:
        db, labels = create_sift_database()
        sift_database = db
        sift_database_labels = labels

    # Segment image and run NN_Sift_Classifier on each character
    for entry in plate_imgs:
        # get data from localization entry
        image = entry[0]
        frame_no = entry[1]

        # plot image
        # plt.figure()
        # plt.subplot(2, 1, 1)
        # plt.imshow(image)

        # treshold image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 4)

        # open morphology
        thresh = morph_open(thresh, True)

        # plot image
        # plt.subplot(2, 1, 2)
        # plt.imshow(thresh)
        # plt.show()

        # segment characters
        char_imgs, dashes = segment_characters(thresh)

        # for im in char_imgs:
        #     im = morph_open(im, False)

        # plot
        # plt.figure()
        # i = 1
        # for char in char_imgs:
        #     plt.subplot(1, 8, i)
        #     plt.imshow(char)
        #     i += 1
        # plt.show()

        # match character images to symbols
        matched_chars = []
        c = None
        for char in char_imgs:
            # use sift to match image
            match, distance = nn_sift_classifier(char, sift_database, sift_database_labels, len(dashes) == 2)
            matched_chars.append(match[0])
            if len(matched_chars) in dashes:
                matched_chars.append("-")
            #     c = None
            # if c is None:
            #     c = match[0] not in number_chars
            # while (c and match[0] in number_chars) or (not c and match[0] not in number_chars):
            #     distance.pop(match)
            #     match = min(distance, key=distance.get)
            # matched_chars.append(match[0])

        # skip entry if empty
        if len(matched_chars) == 0:
            continue

        # compare matches with the currently active recognition chains
        matched_chars = np.array(matched_chars)
        best_chain = -1
        best_chain_diff_size = math.inf

        # find the chain with the lowest amount of differing characters in the last plate
        for c in range(len(recognize_chains)):
            last_plate = recognize_chains[c][-1]
            set_diff = np.setdiff1d(last_plate, matched_chars)
            set_diff_size = len(set_diff)
            if set_diff_size < best_chain_diff_size:
                best_chain = c
                best_chain_diff_size = set_diff_size

        # append to the found chain or start a new chain
        if best_chain_diff_size <= 2:
            # append to the found chain
            recognize_chains[best_chain].append(matched_chars)
        else:
            # start a new chain
            recognize_chains.append([matched_chars])
            chain_metadata.append(frame_no)

    # majority voting
    matches = []
    for c in range(len(recognize_chains)):
        i = 0
        # get votes for character at pos i
        plate = []
        while i in range(8):
            votes = {}
            for chars in recognize_chains[c]:
                if i < len(chars):
                    # vote for current char
                    if chars[i] in votes:
                        votes[chars[i]] += 1
                    else:
                        votes[chars[i]] = 1

            # make sure there were any votes
            if len(votes) == 0:
                break

            # decide character
            voted_char = max(votes, key=votes.get)

            # add to result
            plate.append(voted_char)
            i += 1
        # if len(plate) == 8:
        matches.append("".join([c for c in plate]))

    return matches, chain_metadata
