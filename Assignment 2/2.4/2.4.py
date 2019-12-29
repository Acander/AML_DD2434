import pylab as pb
import numpy as np
from math import pi
from scipy.spatial.distance import cdist
from scipy.stats import norm
from scipy.stats import gamma
from mpl_toolkits.mplot3d import Axes3D

#Related to the observed data
NUMBER_OF_OBSERVATIONS = 10

#Define a true distribution, parameters. Gamma for tau and normal for Xn given tau and mu.
mean = 5
lamda = 15
a = 2
b = 4

a0 = 1
b0 = 1
mean0 = 1
lamda0 = 1

def main():

    tau = sampleFromGammaDistribution(a, b)
    dataSet = sampleFromNormalDistribution(mean, 1/lamda*tau)

    i = 0

    aN, bN, meanN, lamdaN = iterativeInference(np.mean(dataSet))

    approximatePosterior = qPosterior(aN, bN, meanN, lamdaN)

    uMin, uMax, tauMin, tauMax = -5, 5, 0, 20
    # Create meshgrid
    xx, yy = np.mgrid[uMin:uMax:100j, tauMin:tauMax:100j]

    fig = pb.figure(figsize=(8, 8))
    ax = fig.gca()
    ax.set_xlim(uMin, uMax)
    ax.set_ylim(tauMin, tauMax)
    cfset = ax.contourf(xx, yy, approximatePosterior, cmap='coolwarm')
    ax.imshow(np.rot90(approximatePosterior), cmap='coolwarm', extent=[uMin, uMax, tauMin, tauMax])
    cset = ax.contour(xx, yy, approximatePosterior, colors='k')
    ax.clabel(cset, inline=1, fontsize=10)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    pb.title('Approximate Distribution q')

    pb.plot

    truePosterior = posterior(a0, b0, mean0, lamda0, tau, sampleFromNormalDistribution(mean, tau), dataSet)

    fig = pb.figure(figsize=(8, 8))
    ax = fig.gca()
    ax.set_xlim(uMin, uMax)
    ax.set_ylim(tauMin, tauMax)
    cfset = ax.contourf(xx, yy, truePosterior, cmap='coolwarm')
    ax.imshow(np.rot90(truePosterior), cmap='coolwarm', extent=[uMin, uMax, tauMin, tauMax])
    cset = ax.contour(xx, yy, truePosterior, colors='k')
    ax.clabel(cset, inline=1, fontsize=10)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    pb.title('The True Posterior')

def printInferenceResults(aN, bN, meanN, lamdaN):
    print("True parameters:")
    printParameters(a, b, mean, lamda)
    print("Inferred results:")
    printParameters(aN, bN, meanN, lamdaN)

def printParameters(a, b, mean, lamda):
    print("a: ", a, "\tb: ", b, "\tmean: ", mean, "\tlamda: ", lamda)

def posterior(a, b, mean, lamda, tau, u, dataSet):
    return muPrior(mean, lamda*tau)*tauPrior(a, b)*likelihood(dataSet, u, tau)

def likelihood(dataSet, u, tau):
    firstExpression = (tau/2*np.pi)^(NUMBER_OF_OBSERVATIONS/2)
    exponent = (tau/2)*np.sum((dataSet-u)^2)
    return firstExpression*np.exp(exponent)

def qPosterior(aN, bN, meanN, lamdaN):
    return muPrior(meanN, lamdaN)*tauPrior(aN, bN)

def muPrior(mean, precision):
    return norm.pdf(mean, 1/precision)

def tauPrior(a, b):
    return gamma.pdf(a, b)

def sampleFromGammaDistribution(a, b):
    return np.random.gamma(a, b)

def sampleFromNormalDistribution(mean, variance):
    return np.random.normal(mean, variance, NUMBER_OF_OBSERVATIONS)

def expectedValueTau(aN, bN):
    ev = aN/bN
    return ev

def expectedValueMu(observations, meanN, lamdaN):
    squareObservationSum = 0
    for e in enumerate(observations):
        squareObservationSum += e^2
    return (-2*np.sum(observations) + NUMBER_OF_OBSERVATIONS)*meanN + 1 -lamda0*lamdaN^2 + mean0^2 + squareObservationSum

def iterativeInference(meanX, dataSet):
    aN = a0
    bN = b0
    meanN = mean0
    lamdaN = lamda0

    meanN = (lamda0*mean0 + NUMBER_OF_OBSERVATIONS*meanX)/(lamda0 + NUMBER_OF_OBSERVATIONS)
    lamdaN = (lamda0 + NUMBER_OF_OBSERVATIONS)*expectedValueTau(aN, bN)

    aN = a0 + NUMBER_OF_OBSERVATIONS/2
    bN = b0 + 1/2*expectedValueMu(dataSet, meanN, lamdaN)

    return meanN, lamdaN, aN, bN


if __name__ == "__main__":
    main()