import numpy as np

NUMBER_OF_OBSERVATIONS = 10

#Define a true distribution, parameters. Gamma for tau and normal for Xn given tau and mu.
mean = 1
precision = 1/130

a0 = 1
b0 = 1
mean0 = 1
lamda0 = 1

def main():

    plotTrueDistribution()

    #set initial values for approximate distributions
    aN = 0
    bN = 0
    meanN = 0
    lamdaN = 0

    dataSet = sampleFromNormalDistribution(mean, 1/precision)

    i = 0

    q_a, q_b, q_mean, q_lamda = iterativeInference(np.mean(dataSet))

    plotApproximateDistribution()

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

def plotTrueDistribution():


def plotApproximateDistribution(a, b, mean, lamda):

    return

if __name__ == "__main__":
    main()