import numpy as np
import matplotlib.pyplot as plt
import math
import random
from tqdm import tqdm

def RBM():
    #define constants
    xAll = np.array([[-1, -1, -1], [1, -1, 1], [-1, 1, 1], [1, 1, -1], [-1, -1, 1], [1, 1, 1], [1, -1, -1], [-1, 1, -1]])
    nTrials = 2000
    nVisible = 3
    M = np.array([1, 2, 4, 8])
    d_kl = np.zeros(4)
    D_theoretical = np.zeros(4)
    for loop in range(len(M)-1):
        m = M[loop]
        print("Number of Hidden Neurons:", m)
        v = np.zeros(nVisible)  # visible neurons
        h = np.zeros(m)  # hidden neurons
        W = np.random.normal(0, 1, [m, nVisible])  # weight matrix
        thetaH = np.zeros((m)) #hidden layer threshold
        thetaV = np.zeros(nVisible) #visible layer threshold
        nOuter = 3000 #3000
        bh = 0
        bv = 0
        for trial in tqdm(range(nTrials), desc="Training"):
            W, thetaH, thetaV, bh, bv = CD_k(xAll[:4, :], nVisible, m, v, h, W, thetaV, thetaH, bh, bv) #TRAIN CD_k algorithm for trial number of trials

        print("end of training")
        p_b = np.zeros(len(xAll))
        for n in tqdm(range(nOuter), desc="Running Outer"):
            p_b = Outer(xAll, nVisible, m, W, thetaH, thetaV, v, bh, h, bv, nOuter, p_b)
        pData = np.zeros(len(xAll))
        pData[:4] = 1/4

        divergence, theoretical = D_kl(m, nVisible, p_b, pData, d_kl, D_theoretical, loop)
        d_kl[loop] = divergence
        D_theoretical[loop] = theoretical
    print("M: ", M)
    print("d_kl: ", d_kl)
    print("D_theoretical: ", D_theoretical)
    Plotting(d_kl, D_theoretical, M)

    return


def CD_k(x, nVisible, m, v, h, W, thetaV, thetaH, bh, bv):
    dW = np.zeros_like(W) #initialize deltaW
    dThetaH = np.zeros_like(thetaH) #initialize deltaThetaHidden
    dThetaV = np.zeros_like(thetaV) #initialize deltaThetaVisible
    dW, dThetaH, dThetaV, bh, bv = mini_batch(x, nVisible, m, v, h, W, thetaV, thetaH, dW, dThetaH, dThetaV, bh, bv) #run mini batches
    W += dW
    thetaH += dThetaH
    thetaV += dThetaV
    return W, thetaH, thetaV, bh, bv


def mini_batch(x, nVisible, m, v, h, W, thetaV, thetaH, dW, dThetaH, dThetaV, bh, bv):
    nMinibatch = 20 #number of mini-batches #20
    k = 200 #number of steps 200

    for batch in range(nMinibatch):
        randIndex = random.randint(1, len(x)-1)
        v_0 = np.array(x[randIndex, :])
        v = np.copy(v_0)
        bh_0 = LocalFieldHidden(v, W, thetaH)
        bh = np.copy(bh_0)
        for i in range(m):
            h[i] = stochastic_update(bh[i])
        for t in range(k+1):
            bv = LocalFieldVisible(h, W, thetaV)
            for i in range(nVisible):
                v[i] = stochastic_update(bv[i])
            bh = LocalFieldHidden(v, W, thetaH)
            for i in range(m):
                h[i] = stochastic_update(bh[i])

        dW, dThetaH, dThetaV = updateWeights(bh_0, v_0, bh, v, dW, dThetaV, dThetaH)

    return dW, dThetaH, dThetaV, bh, bv


def LocalFieldVisible(input, W, theta):
    b = np.dot(input.T, W) - theta
    return b


def LocalFieldHidden(input, W, theta):
    # print(W)
    # print(input)
    # print(theta)
    b = np.dot(W, input) - theta
    return b


def stochastic_update(b): #stochastic update
    r = random.random()
    p_b = 1/(1 + math.exp(-2 * b))
    if r <= p_b:
        output = 1
    else:
        output = -1
    return output


def updateWeights(bh_0, v_0, bh, v, dW, dThetaV, dThetaH):
    eta = 0.005 #0.005

    dW += eta*(np.outer(np.tanh(bh_0), v_0) - np.outer(np.tanh(bh), v))
    dThetaV += -eta*(v_0 - v)
    dThetaH += -eta*(np.tanh(bh_0)-np.tanh(bh))
    return dW, dThetaH, dThetaV


def D_kl(m, nVisible, p_b, pData, d_kl, D_theoretical, loop):
    D = math.inf
    logPData = np.zeros_like(pData)
    logPBoltzmann = np.zeros_like(pData)
    for j in range(len(pData)):
        if p_b[j] == 0:
            logPBoltzmann[j] = 0
        else:
            logPBoltzmann[j] = math.log(p_b[j])
        if pData[j] == 0:
            logPData[j] = 0
        else:
            logPData[j] = math.log(pData[j])
        # print(d_kl)
        # print(pData[j])
        # print(logPData[j])
        # print(logPBoltzmann[j])
        d_kl[loop] += pData[j]*(logPData[j]-logPBoltzmann[j])

    if m < (2 ** (nVisible - 1) - 1):
        D_new = math.log(2) * (nVisible - math.log(m + 1, 2) - (m + 1) / (2 ** math.log(m+1, 2)))
    else:
        D_new = math.log(2) * 0
    if D_new < D:
        D_theoretical[loop] = D_new

    return d_kl[loop], D_theoretical[loop]

# (xAll, nVisible, m, W, thetaH, thetaV, v, bh, h, bv, nOuter, p_b)
def Outer(x, nVisible, m, W, thetaH, thetaV, v, bh, h, bv, nOuter, p_b):
    #select a random pattern
    randIndex = random.randint(1, len(x)-1)
    v = np.array(x[randIndex, :])
    bh = LocalFieldHidden(v, W, thetaH)
    for i in range(m):
        h[i] = stochastic_update(bh[i])
    nInner = 2000 #2000
    for n in range(nInner):
        p_b = Inner(x, nVisible, m, W, thetaV, thetaH, v, h, nInner, nOuter, p_b, randIndex)
    return p_b


def Inner(x, nVisible, m, W, thetaV, thetaH, v, h, nInner, nOuter, p_b, randIndex):
    bv = LocalFieldVisible(h, W, thetaV)
    for i in range(nVisible):
        v[i] = stochastic_update(bv[i])
    bh = LocalFieldHidden(v, W, thetaH)
    for i in range(m):
        h[i] = stochastic_update(bh[i])
    if np.array_equal(x[randIndex],v):
        p_b[randIndex] += 1/(nInner*nOuter)
    return p_b


def Plotting(d_kl, D_theoretical, M):

    plt.figure()
    plt.title("Kullback-Libler divergence vs. Number of hidden neurons")
    plt.scatter(M, d_kl, label = "D_kl RBM")
    plt.plot(M, D_theoretical, linewidth = 2, color = "blue", label = "D_kl Data")
    plt.legend(loc="upper right")
    plt.xlabel("Number of hidden neurons")
    plt.ylabel("KL Divergence")

    return plt.show()


if __name__ == "__main__":
    RBM()

    print("Run has finished")

