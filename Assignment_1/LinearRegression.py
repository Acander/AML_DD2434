import pylab as pb
import numpy as np
from math import pi
from scipy.spatial.distance import cdist
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D

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

    prior = generate2DPrior()
    """print("Input vector: ")
    print(x)
    print("\n")
    print("Output vector: ")
    print(t)"""

    pb.scatter(x, t)
    pb.plot

    x = np.linspace(-10, 10, 500)
    y = np.linspace(-10, 10, 500)
    X, Y = np.meshgrid(x, y)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
    rv = generate2DPrior()

    fig = pb.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, rv.pdf(pos), cmap='viridis', linewidth=0)
    ax.set_xlabel('w0 axis')
    ax.set_ylabel('w1 axis')
    ax.set_zlabel('Z axis')

    pb.show()


def generate2DPrior():
    return multivariate_normal([0, 0], [[1, 0], [0, 1]])

def generateDataSet2D(w0, w1):
    x = np.random.uniform(-1, 1, 200)
    t = []
    for i, xi in enumerate(x):
        ti = w0 * xi + w1 + np.random.normal(0, 0.2)
        t.append(ti)

    return [x, t]


if __name__ == "__main__":
    main()