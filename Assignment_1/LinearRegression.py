import pylab as pb
import numpy as np
from math import pi
from scipy.spatial.distance import cdist

# To sample from a multivariate Gaussian
# f = np.random.multivariate_normal(mu,K);
# To compute a distance matrix between two sets of vectors
# D = cdist(x1,x2)
# To compute the exponetial of all elements in a matri
# E = np.exp(D)

#Data sets has the the shape [x, t] where x is the input vector and t is the output vector
#Date pairs has same indecies in x and t

def main():
    data = generateDataSet2D(0.5, 1.5)
    x = data[0]
    t = data[1]

    print("Input vector: ")
    print(x)
    print("\n")
    print("Output vector: ")
    print(t)

    plot = pb.scatter(x, t)
    pb.show()

def generateDataSet2D(w0, w1):

    x = np.random.uniform(-1, 1, 200)
    t = []
    for i, xi in enumerate(x):
        ti = w0 * xi + w1 + np.random.normal(0, 0.2)
        t.append(ti)

    return [x, t]


if __name__ == "__main__":
    main()