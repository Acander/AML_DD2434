import numpy as np
import plot as p
from scipy.stats import norm
from scipy.stats import gamma

NUMBER_OF_OBSERVATIONS = 50
INFERENCE_ITERATIONS = 1000

#Define a true distribution, parameters. Gamma for tau and normal for Xn given tau and mu.
mean = 5
lamda = 1
a = 3
b = 4

precisionTrue = a/b

#Initial parameter settings for the inferred distribution
a0 = 0
b0 = 0
mean0 = 0
lamda0 = 0

def main():

    #tau = sampleFromGammaDistribution(a, b)
    print(precisionTrue)
    dataSet = sampleFromNormalDistribution(mean, 1/precisionTrue)
    meanN, lamdaN, aN, bN = iterativeInference(np.mean(dataSet), dataSet)
    print(lamdaN)

    p.plotSetUp(mean, lamda, a, b, INFERENCE_ITERATIONS, NUMBER_OF_OBSERVATIONS)
    p.plotPosterior(mean, precisionTrue, qPosterior, aN, bN, meanN, lamdaN)
    p.plotTruePosterior(mean, precisionTrue, posterior, dataSet, a, b, mean, lamda)
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
    firstExpression = (tau/(2*np.pi)) ** (NUMBER_OF_OBSERVATIONS/2)
    exponent = -(tau/2)*np.sum((dataSet-u)**2)
    likelihood = firstExpression * np.exp(exponent)

    return likelihood

def qPosterior(tauValue, aN, bN, muValue, meanN, lamdaN):
    return muPrior(muValue, meanN, lamdaN)*tauPrior(tauValue, aN, bN)

def muPrior(muValue, mean, precision):
    muValue = norm.pdf(muValue, mean, np.sqrt(1 / precision))
    return muValue

def tauPrior(tauValue, a, b):
    priorValue = gamma.pdf(tauValue, a, loc=0, scale=(1 / b))
    return priorValue

def sampleFromNormalDistribution(mean, variance):
    return np.random.normal(mean, variance, NUMBER_OF_OBSERVATIONS)

def expectedValueTau(aN, bN):
    ev = aN/bN
    return ev

#Used to calculate the expected value function with regard to mu
def expectedValueMu(observations, meanN, lamdaN, mean0, lamda0):
    em2 = 1/lamdaN + meanN**2
    sumDataMu = np.sum(np.square(observations) - 2*meanN*observations + em2)
    sumMu = em2 - 2*mean0*meanN + mean0**2
    finalValue = sumDataMu + lamda0*sumMu
    return finalValue

def settingConstantValues(meanX):
    meanN = (lamda0 * mean0 + NUMBER_OF_OBSERVATIONS * meanX) / (lamda0 + NUMBER_OF_OBSERVATIONS)
    aN = a0 + NUMBER_OF_OBSERVATIONS / 2
    return meanN, aN

def settingIteration(aN, bN, dataSet, meanN, lamdaN):
    bN = b0 + 1 / 2 * expectedValueMu(dataSet, meanN, lamdaN, mean0, lamda0)
    lamdaN = (lamda0 + NUMBER_OF_OBSERVATIONS) * expectedValueTau(aN, bN)

    return lamdaN, bN

def iterativeInference(meanX, dataSet):
    bN = 1
    lamdaN = 1

    meanN, aN = settingConstantValues(meanX)

    i = 0
    while i < INFERENCE_ITERATIONS:
        lamdaN, bN = settingIteration(aN, bN, dataSet, meanN, lamdaN)
        i += 1

    return meanN, lamdaN, aN, bN


if __name__ == "__main__":
    main()