import pylab as pb
import numpy as np
from math import pi
from scipy.spatial.distance import cdist
from scipy.stats import norm
from scipy.stats import gamma
from mpl_toolkits.mplot3d import Axes3D

NUMBER_OF_OBSERVATIONS = 10

#Define a true distribution, parameters. Gamma for tau and normal for Xn given tau and mu.
mean = 1
precision = 1/20

a0 = 1
b0 = 1
mean0 = 1
lamda0 = 1

def main():
    dataSet = sampleFromNormalDistribution(mean, 1/precision)

    i = 0

    aN, bN, meanN, lamdaN = iterativeInference(np.mean(dataSet))

    approximatePosterior = qPosterior(aN, bN, meanN, lamdaN)

    truePosterior = posterior()

def posterior(a, b, mean, lamda):
    tau
    return

def muPrior():
    return

def tauPrior(tau, a, b):
    return gamma(tau, a ,b)

def qPosterior(aN, bN, meanN, lamdaN, u, tau):
    return muPriorQ(meanN, lamdaN, u)*tauPriorQ(aN, bN, tau)

def muPriorQ(meanN, lamdaN, u):
    return norm.pdf(u, meanN, lamdaN)

def tauPriorQ(aN, bN, tau):
    return gamma.pdf(tau, aN, bN)

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