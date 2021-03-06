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
    noise = 1
    numberOfDataPoints = 200
    data = generateDataSet2D(0.5, -1.5, numberOfDataPoints, noise)
    x = data[0]
    t = data[1]

    printRawData(x, t)
    pb.scatter(x, t)
    pb.title("True Model Samples")
    pb.xlabel("x")
    pb.ylabel("t")
    pb.show()


    prior = generate2DPrior()
    sigma = 1
    likelihood = generateLikelihood2D(sigma)

    numberOfTrainingSamples = 7

    sampledIndices = np.random.randint(0, numberOfDataPoints-1, numberOfTrainingSamples)
    dataPointsX = np.array([])
    dataPointsT = np.array([])
    for i in range(numberOfTrainingSamples):
        dataPointsX = np.append(dataPointsX, x[sampledIndices[i]])
        dataPointsT = np.append(dataPointsT, t[sampledIndices[i]])

    dataPoints = [dataPointsX, dataPointsT]
    posterior = generate2DPosteriorFromDataPoints(dataPoints, prior, likelihood)

    sampledData = sampleDataPoints(5, posterior)
    print(sampledData)
    w1, w0 = sampledData.T
    pb.scatter(w1, w0)
    pb.title("Posterior Model Samples")
    pb.xlabel("w1")
    pb.ylabel("w0")
    pb.xlim(-3, 3)
    pb.ylim(-3, 3)
    pb.show()

    #Prep plot for prior and posterior
    x = np.linspace(-3, 3, 500)
    y = np.linspace(-3, 3, 500)
    X, Y = np.meshgrid(x, y)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    #Plot prior
    fig = pb.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, prior.pdf(pos), cmap='viridis', linewidth=0)
    ax.set_xlabel('w1 axis')
    ax.set_ylabel('w0 axis')
    ax.set_zlabel('Z axis')

    #Plot posterior
    fig = pb.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, posterior.pdf(pos), cmap='viridis', linewidth=0)
    ax.set_xlabel('w1 axis')
    ax.set_ylabel('w0 axis')
    ax.set_zlabel('Z axis')

    pb.show()

def sampleDataPoints(numberOfPoints, distribution):
    mean = distribution.mean
    covariance = distribution.cov

    return np.random.multivariate_normal(mean, covariance, numberOfPoints)

def printRawData(x,  t):
    print("Input vector: ")
    print(x)
    print("\n")
    print("Output vector: ")
    print(t)
    print("\n")

def generate2DPrior():
    return multivariate_normal([0, 0], [[1, 0], [0, 1]])

def generateLikelihood2D(sigma):
    return multivariate_normal([0, 0], [[sigma, 0], [0, sigma]])

def generate2DPosteriorFromDataPoints(dataPoints, prior, likelihood):
    xList = dataPoints[0]
    t = dataPoints[1]

    x = np.array([[1, xList[0]]])
    for i in range(xList.size - 1):
        x = np.append(x, [[1, xList[i+1]]], axis=0)

    print(x)
    tau = prior.cov
    sigma = likelihood.cov

    printPriorAndLikelihoodCovariance(tau, sigma)

    covariance = np.linalg.inv(np.dot(np.linalg.inv(sigma), np.dot(x.transpose(), x)) + np.linalg.inv(tau))
    mean = np.dot(np.dot(covariance, np.linalg.inv(sigma)), np.dot(x.transpose(), t))

    printDatapoint(x, t)
    printPosteriorParameters(mean, covariance)
    return multivariate_normal(mean, covariance)

def printPriorAndLikelihoodCovariance(tau, sigma):
    print(tau)
    print(np.linalg.inv(tau))
    print("\n")

    print(sigma)
    print(np.linalg.inv(sigma))
    print("\n")
    print("--------------------------")

def printDatapoint(x, t):
    print(x)
    print(t)
    print("\n")
    print("---------------------------")

def printPosteriorParameters(mean, covariance):
    print(mean)
    print(covariance)
    print("\n")
    print("---------------------------")

def generateDataSet2D(w0, w1, numberOfDataPoints, sigma):
    #x = np.random.uniform(-1, 1, numberOfDataPoints)
    genW = [0.5, -1.5]  # np.random.normal(0, 1, 2)
    x = np.linspace(-1, 1, numberOfDataPoints)
    t = np.array([])
    for i, xi in enumerate(x):
        ti = w0 * xi + w1 + np.random.normal(0, sigma)
        t = np.append(t, ti)

    return [x, t]

if __name__ == "__main__":
    main()