import pylab as pb
import numpy as np

"""
            Plot a meshgrid over a inferred distribution
            :return: null
"""
def plotPosterior(meanTrue, precisionTrue, posteriorFunction, a, b, mean, lamda):
    uList, tauList = createLineSpaceList(meanTrue, precisionTrue)
    M, T = np.meshgrid(uList, tauList)
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
    M, T = np.meshgrid(uList, tauList)
    Z = np.zeros_like(M)

    """print(M)
    print(T)
    print(Z)"""

    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            Z[i][j] = posteriorFunction(tauList[j], a, b, uList[i], mean, lamda, dataSet)

    pb.contour(M, T, Z, 5, colors='blue')

def createLineSpaceList(meanTrue, precisionTrue):
    uList = np.linspace(meanTrue - 3, meanTrue + 3, 100)
    tauList = np.linspace(precisionTrue - 0.9, precisionTrue + 0.9, 100)
    #print(tauList)

    return uList, tauList

def plotSetUp(mean, lamda, a, b):
    custom_lines = [pb.Line2D([0], [0], color="red", lw=4),
                    pb.Line2D([0], [0], color="blue", lw=4)]
    fig, ax = pb.subplots()
    ax.legend(custom_lines, ['Inferred', 'True'])
    pb.xlabel("mean")
    pb.ylabel("precision")
    pb.title("True Posterior and Inferred Posterior, Iterations =" + str(iter) + "\n" + "Prior mu = " + str(
        mean) + ", lambda = " + str(lamda) + ", a = " + str(a) + ", b = " + str(b))

def showPlot():
    pb.show()