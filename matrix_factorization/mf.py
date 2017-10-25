'''
An implementation of matrix factorization
@credit: http://www.quuxlabs.com/wp-content/uploads/2010/09/mf.py_.txt

@author: jame phankosol
--------------------------------------------
@INPUT:
    R     : a matrix to be factorized, dimension N x M
    P     : an initial matrix of dimension N x K
    Q     : an initial matrix of dimension M x K
    K     : the number of latent features
    steps : the maximum number of steps to perform the optimisation
    alpha : the learning rate
    beta  : the regularization parameter
@OUTPUT:
    the final matrices P and Q
'''

import numpy as np


def matrix_factorization(P, Q, R, steps=500, lrate=.0002, beta=.02):
    """
    Knowledge:
                      T
            R1 = P x Q ~ R2

    :param P:
    :param Q:
    :param R:
    :param steps:
    :param lrate:
    :param beta:
    :return:
    """
    n, m = R.shape
    K = P.shape[1]

    Q = Q.T
    for step in range(steps):
        for i in range(n):
            for j in range(m):
                if R[i, j] > 0:
                    rij_pred = np.dot(P[i, :], Q[:, j])
                    eij = R[i, j] - rij_pred  # error
                    for k in range(K):
                        P[i, k] += lrate * (2 * eij * Q[k, j] - beta * P[i, k])
                        Q[k, j] += lrate * (2 * eij * P[i, k] - beta * Q[k, j])
        loss = .0
        for i in range(n):
            for j in range(m):
                if R[i, j] > 0:
                    # error
                    rij_pred = np.dot(P[i, :], Q[:, j])
                    error = np.square(R[i, j] - rij_pred)

                    # regularization
                    tmp = 0
                    for k in range(K):
                        tmp += np.square(P[i, k] + Q[k, j])
                    reg_loss = (beta/2) * tmp

                    loss = error + reg_loss
        if loss < 0.001:
            print('break at step: %d' % (step))
            break
    print('non-break')
    return P, Q.T


if __name__ == '__main__':
    step = 2000
    lrate = .002
    beta = .02

    R = [
        [5, 3, 0, 1],
        [4, 0, 0, 1],
        [1, 1, 0, 5],
        [1, 0, 0, 4],
        [0, 1, 5, 4],
    ]
    R = np.array(R)

    n, m = R.shape  # n is n_user, m is n_item
    k = 3  # n_dim

    P = np.random.rand(n, k)
    Q = np.random.rand(m, k)

    newP, newQ = matrix_factorization(P, Q, R, step, lrate, beta)
    newR = np.matmul(newP, newQ.T)
