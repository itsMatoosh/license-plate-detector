import cv2
import numpy as np
import os

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
    
def NN_SIFT_classifier(filename, database):
    classification_label=-1
    distance = {}
    #TODO: Implement the nearest neighbor classifier
    
    # TODO: Steps: 1- Read the image with the file name given in the input
    image = cv2.imread(filename)      
    
    # TODO: Steps: 2- Convert image to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
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
    
	return []