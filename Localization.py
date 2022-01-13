import cv2
import matplotlib.pyplot as plt
import numpy as np

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


def get_gradient(image):
    """Computes the gradient magnitude and direction of an image."""
    # Sobel gradient in x and y direction
    sobel_kernel_y = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ])
    sobel_kernel_x = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1]
    ])

    # get gradient in x and y directions
    image2 = np.float64(image)
    g_x = cv2.filter2D(image2, -1, sobel_kernel_x)
    g_y = cv2.filter2D(image2, -1, sobel_kernel_y)

    # Gradient magnitude

    # compute the "g" gradient magnitude function.
    g = np.sqrt(g_x ** 2 + g_y ** 2)

    # Gradient orientation
    # compute the "theta" gradient orientation function.
    g_x[g_x == 0] = 0.0001
    theta = np.arctan(g_y / g_x)

    return g, theta


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
    gradient, direction = get_gradient(image_f)

    # Non-maximum suppression - step 3 of Canny
    gradient_thin = non_max_supression(gradient, direction)

    # Double threshold - step 4 of Canny
    edges = apply_thresholds(gradient_thin, lower, upper)

    # Edge running
    result = edge_running(edges)
    return np.uint8(result)


def edge_detection(image: np.ndarray):
    """Detects edges in the given image."""
    lower = max(np.median(image) - 1.3 * np.std(image), 0)
    upper = min(lower + np.std(image) / 6, 255)
    print(lower)
    print(upper)
    kernel_size = 5
    sigma = kernel_size // 3
    return canny(image, kernel_size, sigma, lower, upper)


def hough_accumulator(edge_img):
    """
    Input:
    img : 2D list - represents the edges of the image that we want to extract the lines from
    Output:
    houghAccumulator : 2D list - represents the houghAccumulator
    """

    # Creating the Hough Accumulator
    houghAccumulator = np.zeros((rDim, thetaDim))

    # Implement the main loop/s for caclucating the result of the accumulator
    for x in range(xMax):
        for y in range(yMax):
            # If pixel is empty, continue
            if edge_img[x, y] == 0:
                continue

            # Else loop through all possible thetas and compute the accumulator
            step_amt = (thetaMax - thetaMin) / thetaDim
            for theta_step in range(thetaDim):
                theta = thetaMin + step_amt * theta_step
                r = y * np.cos(theta) + x * np.sin(theta)
                houghAccumulator[int((r + rMax) / (2 * rMax) * rDim), theta_step] += 1

    return houghAccumulator


def hough_transform(image: np.ndarray):
    pass


def plate_detection(image: np.ndarray):
    """Performs localization on the given image"""
    # detect edges on image
    image_edges = edge_detection(image)

    plt.imshow(image_edges, cmap='gray')
    plt.show()

    # perform hough transform on image
    image_hough = None

    plate_imgs = []
    return plate_imgs
