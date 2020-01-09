import numpy as np
import plot as p
from scipy import stats

NUMBER_OF_OBSERVATIONS = 10

#Define a true distribution, parameters. Gamma for tau and normal for Xn given tau and mu.
mean = 5
lamda = 15
a = 2
b = 4

#Initial parameter settings for the inferred distribution
a0 = 1
b0 = 1
mean0 = 1
lamda0 = 1

def main():

    tau = sampleFromGammaDistribution(a, b)
    dataSet = sampleFromNormalDistribution(mean, 1/lamda*tau)
    aN, bN, meanN, lamdaN = iterativeInference(np.mean(dataSet), dataSet)

    p.plotSetUp(mean, lamda, a, b)
    p.plotPosterior(mean, a/b, qPosterior, aN, bN, meanN, lamdaN)
    p.plotTruePosterior(mean, a/b, posterior, dataSet, a, b, mean, lamda)
    p.showPlot()

def printInferenceResults(aN, bN, meanN, lamdaN):
    print("True parameters:")
    printParameters(a, b, mean, lamda)
    print("Inferred results:")
    printParameters(aN, bN, meanN, lamdaN)

def printParameters(a, b, mean, lamda):
    print("a: ", a, "\tb: ", b, "\tmean: ", mean, "\tlamda: ", lamda)

def posterior(tauValue, a, b, muValue, mean, lamda, dataSet):
    return muPrior(muValue, mean, lamda*tauValue)*tauPrior(tauValue, a, b)*likelihood(dataSet, muValue, tauValue)

def likelihood(dataSet, u, tau):
    firstExpression = (tau/2*np.pi) ** (NUMBER_OF_OBSERVATIONS/2)
    exponent = (tau/2)*np.sum((dataSet-u)**2)
    likelihood = firstExpression * np.exp(exponent)

    """print("exponent: ", exponent)
    print("first expression: ", firstExpression)
    print(likelihood)"""

    return likelihood

def qPosterior(tauValue, aN, bN, muValue, meanN, lamdaN):
    return muPrior(muValue, meanN, lamdaN)*tauPrior(tauValue, aN, bN)

def muPrior(muValue, mean, precision):
    #muValue = stats.norm.pdf(muValue, mean, 1 / precision)
    muValue = np.exp(-muValue**2/2)/np.sqrt(2*np.pi)

    #print("MuValue: \t", muValue, "mean: \t", mean, "precision: \t", precision)
    return muValue

def tauPrior(tauValue, a, b):
    tauValue = stats.gamma.pdf(tauValue, a, b)
    #print("tauValue: ", tauValue)
    return tauValue

def sampleFromGammaDistribution(a, b):
    return np.random.gamma(a, b)

def sampleFromNormalDistribution(mean, variance):
    return np.random.normal(mean, variance, NUMBER_OF_OBSERVATIONS)

def expectedValueTau(aN, bN):
    ev = aN/bN
    return ev

def expectedValueMu(observations, meanN, lamdaN):
    squareObservationSum = 0
    for e in range(len(observations)):
        squareObservationSum += observations[e] ** 2
    return (-2*np.sum(observations) + NUMBER_OF_OBSERVATIONS)*meanN + 1 -lamda0*lamdaN ** 2 + mean0**2 + squareObservationSum

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