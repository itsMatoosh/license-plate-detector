import numpy as np
import cv2
import matplotlib.pyplot as plt


def create_A(a, b, N):
    A = np.zeros((N, N))
    for i in range(N):
        A[i, i] = -(2 * a + 6 * b)
        A[i, (i + 1) % N] = A[i, i - 1] = a + 4 * b
        A[i, (i + 2) % N] = A[i, i - 2] = -b
    return A


# TODO: Add parts of the computation of the external edge forces
def create_external_edge_force_gradients_from_img(img, sigma=30, w=1):
    # Apply the guassian filter
    kernel = cv2.getGaussianKernel(6 * sigma, sigma)
    gaussian = cv2.filter2D(img, -1, kernel)
    plt.imshow(gaussian)
    plt.show()
    # Find the gradient in x and y directions
    # Sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    # Sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    # I_x = cv2.filter2D(np.float64(gaussian), -1, Sobel_x)
    # I_y = cv2.filter2D(np.float64(gaussian), -1, Sobel_y)
    I_y, I_x = np.gradient(gaussian)
    # Compute the gradient magnitude
    gradient = np.square(I_x) + np.square(I_y)
    # gradient = np.gradient(gaussian)
    # Normalize the gradient magnitude (crucial step)
    gradient = (gradient - np.min(gradient))
    gradient = gradient / np.max(gradient)
    plt.imshow(gradient)
    plt.show()
    # Gradient of gradient magnitude of the image in x and y directions.
    ggmiy, ggmix = np.gradient(gradient)
    ggmix = ggmix / np.max(ggmix)
    ggmiy = ggmiy / np.max(ggmiy)

    # END OF IMPLEMENTATION

    def fx(x, y):
        # Check bounds.
        x[x < 0] = 0.
        y[y < 0] = 0.

        x[x > img.shape[1] - 1] = img.shape[1] - 1
        y[y > img.shape[0] - 1] = img.shape[0] - 1

        return w * ggmix[(y.round().astype(int), x.round().astype(int))]

    def fy(x, y):
        # Check bounds.
        x[x < 0] = 0.
        y[y < 0] = 0.

        x[x > img.shape[1] - 1] = img.shape[1] - 1
        y[y > img.shape[0] - 1] = img.shape[0] - 1

        return w * ggmiy[(y.round().astype(int), x.round().astype(int))]

    return fx, fy


def iterate_contour(x, y, a, b, fx, fy, gamma=0.1, n_iters=100, return_all=True):
    A = create_A(a, b, x.shape[0])
    print(A[2, :7])
    B = np.linalg.inv(np.eye(x.shape[0]) - gamma * A)
    if return_all:
        snakes = []

    for i in range(n_iters):
        x_ = np.dot(B, x + gamma * fx(x, y))
        y_ = np.dot(B, y + gamma * fy(x, y))
        x, y = x_.copy(), y_.copy()
        if return_all:
            snakes.append((x_.copy(), y_.copy()))
        """if n % 50 == 0:
            plt.imshow(snakes[-1])
            plt.show()"""
    if return_all:
        return snakes
    else:
        return (x, y)


def find_contours(edges, alpha, beta, sigma, gamma, iterations, animate=False):
    # define edges of the system
    center = [int(edges.shape[1] // 2), int(edges.shape[0] // 2)]
    x = center[0] + 1.5 * center[0] * np.cos(np.linspace(0, 2 * np.pi, 500))
    y = center[1] + 1.5 * center[1] * np.sin(np.linspace(0, 2 * np.pi, 500))

    # compute the derivatives
    fx, fy = create_external_edge_force_gradients_from_img(edges, sigma)

    # iterate witth the initial edges to find the outter edges
    res = iterate_contour(x, y, alpha, beta, fx, fy, gamma, iterations)
    if animate:
        return [res]
    return [res[-1]]


# Checking the find contours implementation
def pipeline(image):
    image_edges = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    alpha = 0.1
    beta = 0.9
    iterations = 1000
    sigma = 2
    gamma = 5

    contours_reduced = find_contours(image_edges, alpha, beta, sigma, gamma, iterations, False)

    return contours_reduced