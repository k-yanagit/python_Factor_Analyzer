import numpy as np


def varimax(Phi, gamma = 1.0, max_itr = 100, tol = 1e-6):
    """
    description: Varimax rotation by gradient method.

    arg:
        - Phi: Loadings Matrix
        - gamma: Weight fo gradient
        - max_itr: Maximum number of iterations
        - tol: Tolerance error of change by gradient method

    Return:
        - Phi: Raw loadings matrix
        - R: Rotation matrix
    """
    p,k = Phi.shape
    R = np.eye(k)
    d=0
    for i in range(max_itr):
        d_old = d
        Lambda = np.dot(Phi, R)
        u,s,vh = np.linalg.svd(np.dot(Phi.T,np.asarray(Lambda)**3 - (gamma/p) * np.dot(Lambda, np.diag(np.diag(np.dot(Lambda.T,Lambda))))))
        R = np.dot(u,vh)
        d = np.sum(s)
        if d_old!=0 and d/d_old < 1 + tol: break
    return np.dot(Phi, R)
