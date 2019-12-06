import pylab as pb
import numpy as np
import matplotlib.pyplot as plt
import LinearRegression as lr
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
    numberOfDataPoints = 1000
    noise = 1

    data = lr.generateDataSet2D(0.5, -1.5, numberOfDataPoints, noise)
    #data = generateDataSet()
    x = data[0]
    t = data[1]

    printRawData(x, t)
    pb.scatter(x, t)
    pb.title("True Model Samples")
    pb.xlabel("x")
    pb.ylabel("t")
    pb.show()

    sigma = 1
    l = 1

    mean, covariance = generateGPPrior(x, sigma, l)
    print(mean)
    print(covariance)
    samples = np.random.multivariate_normal(mean, covariance, 10)
    print(samples)

    plotCurves(samples)

def plotCurves(samples):
    #pb.plot(samples)
    x = np.arange(len(samples[0]))
    for y in samples:
        plt.plot(x, y)
    plt.show()

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

def generateGPPrior(x, sigma, l):
    gramKernel = kernel(x, sigma, l)
    #print(gramKernel)
    #print(np.zeros(len(x)))
    return np.zeros(len(x)), gramKernel

#Squared Exponential covariance function
def kernel(x, sigma, l):
    gram = []
    for j in range(len(x)):
        xdiff = (x[j] - x)
        #print(xdiff)
        exp = xdiff*xdiff/(l*l)
        #print(exp)
        gramRow = sigma*sigma*np.exp(- exp)
        gram.append(gramRow)
    return np.array(gram)

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



def generateDataSet():
    x = np.array([-4, -3, -2, -1, 0, 2, 3, 5])
    t = np.array([])
    for i, xi in enumerate(x):
        ti = (2 + (0.5 * xi) * (0.5 * xi)) * np.sin(3 * xi) + np.random.normal(0, 0.3)
        t = np.append(t, ti)

    return [x, t]


if __name__ == "__main__":
    main()