import pylab as pb
import numpy as np
from math import pi
from scipy.spatial.distance import cdist
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

# To sample from a multivariate Gaussian
# f = np.random.multivariate_normal(mu,K);
# To compute a distance matrix between two sets of vectors
# D = cdist(x1,x2)
# To compute the exponetial of all elements in a matri
# E = np.exp(D)

#Data sets has the the shape [x, t] where x is the input vector and t is the output vector
#Date pairs has same indecies in x and t

def main():
    data = generateDataSet()
    x = data[0][:, np.newaxis]
    t = data[1]

    pb.scatter(x, t)
    pb.title("True Model Samples")
    pb.xlabel("x")
    pb.ylabel("t")

    sigma = 0.2
    lengthScaleBounds = (1e-1, 10)
    print("Values:")

    lengthScale = []

    for i in range(4):
        lengthScale.append(np.random.uniform(0, 1))

    for i in range(1):
        for j in range(4):
            print(lengthScale[j])

            #Creating a prior
            gp = generateGaussianPrior(sigma, lengthScale[j]*np.sqrt(1/2), lengthScaleBounds)

            # Plot prior
            pb.figure(figsize=(8, 8))
            pb.subplot(2, 1, 1)
            X_ = np.linspace(0, 5, 100)
            y_mean, y_std = gp.predict(X_[:, np.newaxis], return_std=True)
            pb.plot(X_, y_mean, 'k', lw=3, zorder=9)
            pb.fill_between(X_, y_mean - y_std, y_mean + y_std,
                             alpha=0.2, color='k')
            y_samples = gp.sample_y(X_[:, np.newaxis], 10)
            pb.plot(X_, y_samples, lw=1)
            pb.xlim(0, 5)
            pb.ylim(-3, 3)
            pb.title("Prior", fontsize=12)

            #Creating a posterior
            gp.fit(x, t)

            # Plot posterior
            pb.subplot(2, 1, 2)
            X_ = np.linspace(0, 5, 100)
            t_mean, t_std = gp.predict(X_[:, np.newaxis], return_std=True)
            pb.plot(X_, t_mean, 'k', lw=3, zorder=9)
            pb.fill_between(X_, t_mean - t_std, t_mean + t_std,
                             alpha=0.2, color='k')

            y_samples = gp.sample_y(X_[:, np.newaxis], 10)
            pb.plot(X_, y_samples, lw=1)
            pb.scatter(x[:, 0], t, c='r', s=50, zorder=10, edgecolors=(0, 0, 0))
            pb.xlim(0, 5)
            pb.ylim(-3, 3)
            pb.title("Posterior")
            pb.tight_layout()

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

def generateGaussianPrior(sigma, lengthScale, lengthScaleBounds):
    return GaussianProcessRegressor(kernel=sigma*RBF(length_scale=lengthScale, length_scale_bounds=lengthScaleBounds))

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
        ti = (2 + (0.5*xi)*(0.5*xi))*np.sin(3*xi) + np.random.normal(0, 0.3)
        t = np.append(t, ti)

    return [x, t]


if __name__ == "__main__":
    main()