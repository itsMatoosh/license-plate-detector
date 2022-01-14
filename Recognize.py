import cv2
import numpy as np
import os
import scipy

def sift_descriptor(image):
    result = np.zeros(0)
    # Resizes image to 16M * 16M where M is an integer
    """
    msize = np.max(image.shape)
    fsize = (16-(msize%16)) + msize
    fimage = cv2.resize(image, (fsize, fsize), interpolation = cv2.INTER_AREA)"""
    windowsize = 4
    I_y, I_x = np.gradient(image)
    mag = np.sqrt(np.square(I_x) + np.square(I_y))
    direction = np.arctan2(I_y, I_x)
    # Iterate over every pixel
    for x in range(4):
        for y in range(4):
            submag = mag[int(x*4):int((x+1)*4), int(y*4):int((y+1)*4)].flatten()
            subdir = direction[int(x*4):int((x+1)*4), int(y*4):int((y+1)*4)].flatten()
            chist = np.zeros(8)
            for index, b in enumerate(subdir):
                subhist = np.histogram(b, bins = 8, range = (0, 2*np.pi))[0].astype(np.float64)
                subhist[subhist == 1] += submag[index]
                chist += subhist
            result = np.append(result, chist)
            
    assert len(result) == 128
    return np.array(result)
    
    
def NN_SIFT_classifier(image, database):
    classification_label=-1
    distance = {}
    #TODO: Implement the nearest neighbor classifier
    
    # TODO: Steps: 1- Read the image with the file name given in the input
    #image = cv2.imread(filename)      
    
    # TODO: Steps: 2- Convert image to grayscale
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # TODO: Steps: 3- resize the image if needed to fit the training set
    image = cv2.resize(image, (16, 16))
    
    # TODO: Steps: 4- find the SIFT descriptor of the test image
    sift = sift_descriptor(image)
    
    # TODO: Steps: 5- Measure the similarity between the test image descriptor & all the database images' descriptors
    for key in database.keys():
        other_sift = database[key]
        distance[key] = compare_euclidean_norm(sift, other_sift)
    
    # TODO: Steps: 6- Sort the labels based on the similarity  
    distance = dict(sorted(distance.items(), key=lambda item: item[1]))
   
    # TODO: Steps: 7- print the label & the distances of the sorted list
    # Commented this out because its a lot of clutter
    #for key in distance.keys():
    #    print("{} -> {}".format(key, distance[key]))
    
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
    background = 256 - foreground
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
            image = cv2.imread("data/SameSizeLetters" + fname)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (16, 16))
            sift = sift_descriptor(image)
            database[filename] = sift
    
    for fname in os.listdir("data/SameSizeNumbers"):
        if fname.endswith('.png') or fname.endswith('.jpg') or fname.endswith('.bmp'):
            image = cv2.imread("data/SameSizeNumbers" + fname)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (16, 16))
            sift = sift_descriptor(image)
            database[filename] = sift
            
    # TODO: Segment image and run NN_Sift_Classifier on each character
    
    for image in plate_imgs:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = isodata_thresholding(gray)
        start = False
        startIndex = 0
        imgs = []
        for x in range(thresh.shape[1]):
            col = thresh[:, x]
            try:
                prop = np.bincount(col)[256]/len(col)
            except:
                prop = 0
            if prop > 0.09 and not start:
                startIndex = x
                start = True
            if prop < 0.09 and start:
                start = False
                if x - startIndex > 6:
                    imgs.append(thresh[:, startIndex:x])
        
        matches = []
        for char in imgs:
            match, distance = NN_SIFT_classifier(char, database)
            matches.append(match[0])
        
        res.append(matches)
    
    return res