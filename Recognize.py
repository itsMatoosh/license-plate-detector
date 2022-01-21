import cv2
import numpy as np
import os
import scipy
import matplotlib.pyplot as plt

from tools import gradient


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

    # crop image to remove 0s around
    nonzero = np.argwhere(image > 200)[:, 0]
    y_start = np.min(nonzero)
    y_end = np.max(nonzero)
    image = image[y_start:y_end+1]

    # resize to be height 85
    new_h = 85
    scale = new_h / image.shape[0]
    new_w = int(image.shape[1] * scale)
    image = cv2.resize(image, (new_w, new_h))

    # adjust to be the same size as database picture
    canvas = np.zeros((85, 100))
    if (new_w > 100):
        new_w = 100
    canvas[:, 0:new_w] = image[:, 0:new_w]

    distance = {}
    # Implement the nearest neighbor classifier

    # Steps: 1- Read the image with the file name given in the input
    # image = cv2.imread(filename)

    # Steps: 2- Convert image to grayscale
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Steps: 3- resize the image if needed to fit the training set
    image = cv2.resize(canvas, (16, 16))

    # Steps: 4- find the SIFT descriptor of the test image
    sift = sift_descriptor(image)

    # Steps: 5- Measure the similarity between the test image descriptor & all the database images' descriptors
    for key in database:
        distance[key] = compare_euclidean_norm(sift, database[key])

    # Steps: 6- Sort the labels based on the similarity
    distance = dict(sorted(distance.items(), key=lambda x: x[1]))

    # Steps: 7- print the label & the distances of the sorted list
    print("new char")
    for key in distance:
        print(key + ": " + str(distance[key]))

    # Return the label of the classification with the minimum distance & the disatance 
    return min(distance, key=distance.get), distance
    
    
def isodata_thresholding(image, epsilon = 2):
    # Compute the histogram and set up variables
    hist = np.array(cv2.calcHist([image], [0], None, [256], [0, 256])).flatten()
    tau = np.random.randint(hist.nonzero()[0][0], 256 - hist[::-1].nonzero()[0][0])
    old_tau = -2*epsilon
    
    # Iterations of the isodata thresholding algorithm
    while(abs(tau - old_tau) >= epsilon):
        #TODO Calculate m1
        m1 = 0
        for i in range(tau):
            m1 += i*hist[i]
        m1 = m1 / hist[:tau].sum()
        #TODO Calculate m2
        m2 = 0
        for i in range(tau, 256):
            m2 += i * hist[i]
        m2 = m2 / hist[tau:].sum()
        
        #TODO Calculate new tau
        old_tau = tau
        tau = int((m1 + m2)/2)
    
    #TODO Threshold the image based on last tau
    ret, foreground = cv2.threshold(image, tau, 255, cv2.THRESH_BINARY)
    background = 255 - foreground
    return background

"""
In this file, you will define your own segment_and_recognize function.
To do:
	1. Segment the plates character by character
	2. Compute the distances between character images and reference character images(in the folder of 'SameSizeLetters' and 'SameSizeNumbers')
	3. Recognize the character by comparing the distances
Inputs:(One)
	1. plate_imgs: cropped plate images by Localization.plate_detection function
	type: list, each element in 'plate_imgs' is the cropped image(Numpy array)
Outputs:(One)
	1. recognized_plates: recognized plate characters
	type: list, each element in recognized_plates is a list of string(Hints: the element may be None type)
Hints:
	You may need to define other functions.
"""
def segment_and_recognize(plate_imgs):
    database = {}
    res = []
    for fname in os.listdir("data/SameSizeLetters"):
        if fname.endswith('.png') or fname.endswith('.jpg') or fname.endswith('.bmp'):
            image = cv2.imread("data/SameSizeLetters/" + fname)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (16, 16))
            sift = sift_descriptor(image)
            database[fname] = sift
    
    for fname in os.listdir("data/SameSizeNumbers"):
        if fname.endswith('.png') or fname.endswith('.jpg') or fname.endswith('.bmp'):
            image = cv2.imread("data/SameSizeNumbers/" + fname)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (16, 16))
            sift = sift_descriptor(image)
            database[fname] = sift
            
    # TODO: Segment image and run NN_Sift_Classifier on each character
    for image in plate_imgs:
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = isodata_thresholding(gray)
        start = False
        startIndex = 0
        imgs = []
        reqWidth = thresh.shape[1] * 0.02
        for x in range(thresh.shape[1]):
            col = thresh[:, x]
            try:
                prop = np.bincount(col)[255]/len(col)
            except:
                prop = 0
            if prop > 0.1 and not start:
                startIndex = x
                start = True
            if prop < 0.1 and start:
                start = False
                if x - startIndex > reqWidth:
                    imgs.append(thresh[:, startIndex:x])

        matches = []
        letter = None
        for char in imgs:
            match, distance = NN_SIFT_classifier(char, database)
            if match[0] == '-':
                letter = None
                matches.append('-')
                continue
            if letter == None:
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
        
        res.append(matches)
    
    return res