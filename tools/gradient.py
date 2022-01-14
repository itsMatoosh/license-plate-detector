import cv2
import numpy as np


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
