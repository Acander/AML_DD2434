import pylab as pb
import numpy as np

"""
            Plot a meshgrid over a inferred distribution
            :return: null
"""
def plotPosterior(meanTrue, precisionTrue, posteriorFunction, a, b, mean, lamda):
    uList, tauList = createLineSpaceList(meanTrue, precisionTrue)
    M, T = np.meshgrid(uList, tauList, indexing="ij")
    Z = np.zeros_like(M)

    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            Z[i][j] = posteriorFunction(tauList[j], a, b, uList[i], mean, lamda)

    #print(Z)
    pb.contour(M, T, Z, 5, colors='red')

"""
            Plot a meshgrid over the true distribution 
            :return: null
"""
def plotTruePosterior(meanTrue, precisionTrue, posteriorFunction, dataSet, a, b, mean, lamda):
    uList, tauList = createLineSpaceList(meanTrue, precisionTrue)
    M, T = np.meshgrid(uList, tauList, indexing="ij")
    Z = np.zeros_like(M)

    """print(M)
    print(T)
    print(Z)"""

    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            Z[i][j] = posteriorFunction(tauList[j], a, b, uList[i], mean, lamda, dataSet)

    pb.contour(M, T, Z, 5, colors='blue')

def createLineSpaceList(meanTrue, precisionTrue):
    uList = np.linspace(meanTrue - 0.5, meanTrue + 0.5, 100)
    tauList = np.linspace(precisionTrue - 0.75, precisionTrue + 0.75, 100)
    #print(tauList)

    return uList, tauList

def plotSetUp(mean, lamda, a, b, VI_iter, N_obs):
    custom_lines = [pb.Line2D([0], [0], color="red", lw=4),
                    pb.Line2D([0], [0], color="blue", lw=4)]
    fig, ax = pb.subplots()
    ax.legend(custom_lines, ['Inferred', 'True'])
    pb.xlabel("Mean")
    pb.ylabel("Precision")
    pb.title("Posteriors, Iterations: " + str(VI_iter) + ", Observations: " + str(N_obs) + "\n" + "True Prior Params: mu = " +
             str(mean) + ", lambda = " + str(lamda) + ", a = " + str(a) + ", b = " + str(b))

def showPlot():
    pb.show()