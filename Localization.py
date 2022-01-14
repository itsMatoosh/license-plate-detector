import math

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters

from tools import gradient
from tools import contours

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


def non_max_supression(gradient, direction, epsilon=0.000001):
    """Performs non-max suppression on a given image."""

    # Create empty matrix of image shape
    result = np.zeros(gradient.shape)

    # Loop over every pixel
    for x in range(gradient.shape[0] - 3):
        for y in range(gradient.shape[1] - 3):
            # Check the direction of the pixel
            d = direction[x + 1, y + 1]

            # Based on direction find whether the value is to be saved or not
            if np.abs(d) >= 3 * np.pi / 8:
                if max(gradient[x, y + 1], gradient[x + 1, y + 1],
                       gradient[x + 2, y + 1]) - gradient[x + 1, y + 1] < epsilon:
                    result[x + 1, y + 1] = gradient[x + 1, y + 1]
            elif 3 * np.pi / 8 > d >= np.pi / 8:
                if max(gradient[x, y + 2], gradient[x + 1, y + 1],
                       gradient[x + 2, y]) - gradient[x + 1, y + 1] < epsilon:
                    result[x + 1, y + 1] = gradient[x + 1, y + 1]
            elif np.pi / 8 > d >= -np.pi / 8:
                if max(gradient[x + 1, y], gradient[x + 1, y + 1],
                       gradient[x + 1, y + 2]) - gradient[x + 1, y + 1] < epsilon:
                    result[x + 1, y + 1] = gradient[x + 1, y + 1]
            elif -np.pi / 8 > d >= -3 * np.pi / 8:
                if max(gradient[x, y], gradient[x + 1, y + 1],
                       gradient[x + 2, y + 2]) - gradient[x + 1, y + 1] < epsilon:
                    result[x + 1, y + 1] = gradient[x + 1, y + 1]

    # Return matrix with only max values
    return result


# The weak and strong arguments are numbers we use to mark whether a pixel contains a weak or a strong edge
def apply_thresholds(edges_supressed, lower, upper, weak=127, strong=255):
    # Create an empty result array with same shape as edges
    result = np.zeros(edges_supressed.shape)

    # Loop over the found edges
    for x in range(edges_supressed.shape[0]):
        for y in range(edges_supressed.shape[1]):
            edge = edges_supressed[x, y]

            # If edge value is >= than the upper limit, label it as strong
            if edge >= upper:
                result[x, y] = strong
            # If edge value is within (lower, upper), label it as weak
            elif edge < upper and edge > lower:
                result[x, y] = weak
            # Otherwise, label it as 0

    return result


# The weak and strong arguments are numbers we use to mark whether a pixel contains a weak or a strong edge
def edge_running(edges, weak=127, strong=255):
    # Create an empty result array with same shape as edges
    result = np.zeros(edges.shape)

    # Loop over edges
    for x in range(edges.shape[0] - 3):
        for y in range(edges.shape[1] - 3):
            edge = edges[x + 1, y + 1]
            # If the edges is weak, check if there is a maximal value withing its closest neighbours
            if edge == weak:
                # If there is a maximum, mark it as a strong edge, if not then discard the edge
                maximum = np.max(edges[x:x + 3, y:y + 3])
                if maximum == strong:
                    result[x + 1, y + 1] = strong

                    # If the edge is already strong, mark it as strong
            elif edge == strong:
                result[x + 1, y + 1] = strong

    return result


def canny(image: np.ndarray, size, sigma, lower, upper):
    """Performs canny edge detection on an image."""
    # Making the image greyscale
    image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Noise reduction using Gaussian kernel - step 1 of Canny
    kernel = cv2.getGaussianKernel(size, sigma)
    image_f = cv2.filter2D(image_grey, -1, kernel)

    # Gradient calculation - step 2 of Canny
    gradient, direction = gradient.get_gradient(image_f)

    # Non-maximum suppression - step 3 of Canny
    gradient_thin = non_max_supression(gradient, direction)

    # Double threshold - step 4 of Canny
    edges = apply_thresholds(gradient_thin, lower, upper)

    # Edge running
    result = edge_running(edges)
    return np.uint8(result)


def morphology_close(image: np.ndarray):
    """Performs the morphological close operation on an image."""
    structuring_element = np.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]
    ], dtype=np.uint8)
    img_dilated = cv2.dilate(image, structuring_element)
    return cv2.erode(img_dilated, structuring_element)


def edge_detection(image: np.ndarray):
    """Detects edges in the given image."""
    lower = 30
    upper = 60
    kernel_size = 5
    sigma = kernel_size // 3
    return canny(image, kernel_size, sigma, lower, upper)


def hough_accumulator(edge_img, r_dim, theta_dim, theta_min, theta_max, r_max):
    """
    Input:
    img : 2D list - represents the edges of the image that we want to extract the lines from
    Output:
    houghAccumulator : 2D list - represents the houghAccumulator
    """

    # Creating the Hough Accumulator
    accumulator = np.zeros((r_dim, theta_dim))

    # dimensions of edge img
    H = edge_img.shape[0]
    W = edge_img.shape[1]

    # Implement the main loop/s for caclucating the result of the accumulator
    for y in range(H):
        for x in range(W):
            # If pixel is empty, continue
            if edge_img[y, x] == 0:
                continue

            # Else loop through all possible thetas and compute the accumulator
            step_amt = (theta_max - theta_min) / (theta_dim - 1)
            for i in range(theta_dim):
                theta = theta_min + step_amt * i
                r = x * np.cos(theta) + y * np.sin(theta)
                accumulator[int(r / r_max * r_dim), i] += 1

    return accumulator


# Input:
# houghAccumulator : 2D list - represents the houghAccumulator
# neighborhoodSize : int - represent the size of the neighbours to consider
# threshold: int - represent the minimum difference between the maxima and minima for the window to be considered
# Output:
# theta : list - contains all horizontal coordinates of the local extremas that were found
# rho : list - contains all vertical coordinates of the local extremas that were found
def find_interest_points(h_accumulator, neighborhood_size, threshold):
    # Find local maximas
    data_max = filters.maximum_filter(h_accumulator, neighborhood_size)
    maxima = (h_accumulator == data_max)

    # Find local minimas
    data_min = filters.minimum_filter(h_accumulator, neighborhood_size)

    # Preserve difference between max and min only if it is greater than the threshold
    diff = ((data_max - data_min) > threshold)
    maxima[diff == 0] = 0

    # Identify the maxima regions in the image.
    labeled, num_objects = ndimage.label(maxima)
    slices = ndimage.find_objects(labeled)

    # Calculate the central point of the maxima regions
    theta, rho = [], []
    for dy, dx in slices:
        # Append the central x coordinate to theta
        # Here you can get the starting horizontal coordinate with dx.start and ending point with dx.stop
        theta.append((dx.start + dx.stop) / 2)

        # Append the central y coordinate to rho
        # Here you can get the starting vertical coordinate with dy.start and ending point with dy.stop
        rho.append((dy.start + dy.stop) / 2)

    return theta, rho


# Input:
# theta : list - represents horizontal coordinate of local extremas
# rho : list - represents vertical coordinate of loacl extremas
# Output:
# lines : list - contains lists of points thar represent all the points a specific line traverses
def get_lines(edge_image, theta, rho, r_dim, theta_dim, r_max, theta_max, x_max, y_max):
    # Array to contain all the lines
    lines = []

    # Iterate over you previous results and extract the lines that were chosen
    for i in range(len(theta)):
        # get rho and theta for this line
        t = theta[i] / theta_dim * theta_max
        r = rho[i] / r_dim * r_max

        # go through each pixel
        points = []
        for x in range(x_max):
            y = int(-(np.cos(t) / np.sin(t)) * x + (r/np.sin(t)))
            if 0 <= y < y_max:
                points.append([x, y])

        # add to lines
        lines.append(points)

    return lines


def line_slope(line):
    """Gets the slope of a given line."""
    if len(line) < 2:
        return -1
    p_start = line[0]
    p_end = line[-1]
    return (p_end[1] - p_start[1]) / (p_end[0] - p_start[0])

# Input:
# line1 : list - contains 2 points, the starting point at index 0 and the ending point at index 1
# line2 : list - contains 2 points, the starting point at index 0 and the ending point at index 1
# Output:
# x, y : int, int - coordinates of the intersection between the two lines. -1, -1 if no intersection was found
def line_intersection(line1, line2):
    # get first and last points
    p_low_1 = line1[0]
    p_high_1 = line1[-1]
    p_low_2 = line2[0]
    p_high_2 = line2[-1]

    # get slopes of both lines
    m1 = line_slope(line1)
    m2 = line_slope(line2)

    # check div by 0
    if m1 - m2 == 0:
        return -1, -1

    # get intersection point x and y
    x = (p_low_2[1] - p_low_1[1] + m1 * p_low_1[0] - m2 * p_low_2[0]) / (m1 - m2)

    # check bounds
    if x < p_low_1[0] or x > p_high_1[0] or x < p_low_2[0] or x > p_high_2[0]:
        return -1, -1

    y = m1 * (x - p_low_1[0]) + p_low_1[1]

    return int(x), int(y)


def find_intersections(lines, x_max, y_max):
    intersectionsX = []
    intersectionsY = []

    for l1 in lines:
        for l2 in lines:
            if (l1 != l2 and l1 != [] and l2 != []):
                A = [l1[0][0], l1[0][1]]
                B = [l1[-1][0], l1[-1][1]]
                C = [l2[0][0], l2[0][1]]
                D = [l2[-1][0], l2[-1][1]]

                intersect = line_intersection((A, B), (C, D))

                if (intersect[0] >= 0 and intersect[1] >= 0 and intersect[0] <= x_max and intersect[1] <= y_max):
                    intersectionsX.append(intersect[0])
                    intersectionsY.append(intersect[1])
    return intersectionsX, intersectionsY


def find_license_intersections(edge_image: np.ndarray):
    """Computes the hough transform of an image.
    Input must be an edge image."""
    # Getting image dimensions
    img_shape = edge_image.shape
    y_max = img_shape[0]
    x_max = img_shape[1]

    # Setting angle boundaries
    theta_max = 1.0 * math.pi
    theta_min = 0.0 * math.pi

    # Setting rho boundaries
    r_max = math.hypot(x_max, y_max)

    # Setting step ranges- Quantization of parameter space (you can try different values here)
    r_dim = 200
    theta_dim = 200

    # compute hough accumulator
    h_accumulator = hough_accumulator(edge_image, r_dim, theta_dim, theta_min, theta_max, r_max)

    # find interest points in the accumulator
    neighborhood_size = 12
    threshold = 20
    theta, rho = find_interest_points(h_accumulator, neighborhood_size, threshold)

    # find lines in the image
    lines = get_lines(edge_image, theta, rho, r_dim, theta_dim, r_max, theta_max, x_max, y_max)

    # filter lines for more vertical and horizontal lines
    lines_filtered = []
    for line in lines:
        if len(line) < 2:
            continue
        m = abs(line_slope(line))
        if m < 0.1 or m > 5:
            lines_filtered.append(line)

    # # Plot the original image
    # Using subplots to be able to plot the lines above it
    fig, ax = plt.subplots()
    plt.figure(figsize=(20, 10))
    ax.imshow(edge_image, cmap='gray')

    for line in lines_filtered:
        ax.plot([point[0] for point in line], [point[1] for point in line], linewidth=3)

    # Showing the final plot with all the lines
    plt.show()

    # find intersections between lines
    return find_intersections(lines_filtered, x_max, y_max)


def preprocess_image(image: np.ndarray):
    """Preprocesses the input image"""
    # Convert to HSI/HSV
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define color range
    color_min = np.array([16, 50, 80])
    color_max = np.array([45, 255, 255])

    # Segment only the selected color from the image and leave out all the rest (apply a mask)
    mask = cv2.inRange(image_hsv, color_min, color_max)

    # Plot the masked image (where only the selected color is visible)
    image_masked = cv2.bitwise_and(image_hsv, image_hsv, mask=mask)

    return cv2.cvtColor(image_masked, cv2.COLOR_HSV2RGB)


def plate_detection(image: np.ndarray):
    """Performs localization on the given image"""
    # show initial image
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # preprocess image
    image_processed = preprocess_image(image)
    plt.imshow(image_processed)
    plt.show()

    # detect edges on image
    image_edges = edge_detection(image_processed)

    # perform hough transform on image
    intersections_x, intersections_y = find_license_intersections(image_edges)

    # Plot the result
    plt.figure(figsize=(10, 10))
    plt.imshow(image_rgb)
    plt.plot(intersections_x, intersections_y, 'ro')
    plt.title('Normal Space')
    plt.show()

    plate_imgs = []
    return plate_imgs