import numpy as np

NUMBER_OF_ITERATIONS = 10
NUMBER_OF_DATA_SAMPLES = 100

#Define a true distribution, parameters. Gamma for tau and normal for Xn given tau and mu.
mean = 1
precision = 1/130

a0 = 0
b0 = 0
mean0 = 0
lamda0 = 0

def main():

    plotTrueDistribution(a0, b0, mean0, lamda0)

    #set initial values for approximate distributions
    aN = 0
    bN = 0
    meanN = 0
    lamdaN = 0

    dataSet = sampleFromNormalDistribution(mean, 1/precision)

    i = 0

    q_a, q_b, q_mean, q_lamda = iterativeInference(np.mean(dataSet))

    i += 1

    plotApproximateDistribution(a, b, mean, lamda)

def sampleFromGammaDistribution(a, b):
    return np.random.gamma(a, b)

def sampleFromNormalDistribution(mean, variance):
    return np.random.normal(mean, variance, NUMBER_OF_DATA_SAMPLES)

def expectedValueTau():
    ev = a0/b0
    return ev

def expectedValueMu

def iterativeInference(meanX, dataSet):
    meanN = (lamda0*mean0 + NUMBER_OF_ITERATIONS*meanX)/(lamda0 + NUMBER_OF_ITERATIONS)
    lamdaN = (lamda0 + NUMBER_OF_ITERATIONS)*expectedValueTau()

    aN = a0 + NUMBER_OF_ITERATIONS/2
    bN = b0 + 1/2*

def plotApproximateDistribution(a, b, mean, lamda):

    return

if __name__ == "__main__":
    main()