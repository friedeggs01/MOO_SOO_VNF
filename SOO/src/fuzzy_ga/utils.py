import numpy as np

def triangle_mf(x, params):
    """
        x: array 1D
    """
    a, b, c = np.r_[params]
    y = np.zeros(len(x))

    # Left side
    if a != b:
        idx = np.nonzero(np.logical_and(a < x, x < b))[0]
        y[idx] = (x[idx] - a) / float(b - a)

    # Right side
    if b != c:
        idx = np.nonzero(np.logical_and(b < x, x < c))[0]
        y[idx] = (c - x[idx]) / float(c - b)

    idx = np.nonzero(x == b)
    y[idx] = 1
    return y

def inverse_triangle_mf(y, params): # return center of the area
    a, b, c = np.r_[params]
    left = (b-a)*y + a
    right = c - (c-b)*y
    return (left + right) / 2

def positive_linear_mf(x, params):
    a, b = np.r_[params]
    y = np.zeros(len(x))

    # x <= a -> y = 0

    # a < x < b -> linear
    idx = np.nonzero(np.logical_and(a < x, x < b))[0]
    y[idx] = (x[idx] - a) / float(b-a)

    # x >= b -> y = 1
    idx = np.nonzero(x >= b)[0]
    y[idx] = 1

    return y

def inverse_positive_linear_mf(y, params): # return center of the area
    a, b = np.r_[params]
    left = (b-a)*y + a
    right = 1
    return  (left+right)/2

def negative_linear_mf(x, params):
    a, b = np.r_[params]
    y = np.ones(len(x))

    # x <= a -> y = 1

    # a < x < b -> linear
    idx = np.nonzero(np.logical_and(a < x, x < b))[0]
    y[idx] = (b-x[idx]) / float(b-a)

    # x >= b -> y = 0
    idx = np.nonzero(x >= b)[0]
    y[idx] = 0

    return y

def inverse_negative_linear_mf(y, params): # return center of the area
    a, b = np.r_[params]
    left = 0
    right = b - (b-a)*y
    return  (left+right)/2

if __name__ == "__main__":
    print(triangle_mf(np.array([0.1, 0.3, 0.5, 0.7, 0.9]), np.array([0.1, 0.5, 0.9])))
    print(positive_linear_mf(np.array([0.1, 0.3, 0.5, 0.7, 0.9]), np.array([0.5, 0.9])))
    print(negative_linear_mf(np.array([0.1, 0.3, 0.5, 0.7, 0.9]), np.array([0.1, 0.5])))
    