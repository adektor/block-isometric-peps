import numpy as np
import math
from ncon import ncon
import itertools as itt
from matplotlib import pyplot as plt

def fidelity(Gamma, sigma, sigma_t):
    psi_phi = ncon([sigma, Gamma, sigma_t], ((1, 2), (1, 2, 3, 4), (3, 4)))
    phi_psi = ncon([sigma_t, Gamma, sigma], ((1, 2), (1, 2, 3, 4), (3, 4)))
    phi_phi = ncon([sigma_t, Gamma, sigma_t], ((1, 2), (1, 2, 3, 4), (3, 4)))
    psi_psi = ncon([sigma, Gamma, sigma], ((1, 2), (1, 2, 3, 4), (3, 4)))
    f = (phi_psi * psi_phi) / (phi_phi * psi_psi)
    
    assert math.isclose(psi_phi, phi_psi, rel_tol=1e-09, abs_tol=0.0)

    # print("norm of psi: {0}".format(psi_psi))
    # print("norm of phi: {0}".format(phi_phi))
    # print("overlap    : {0}".format(psi_phi))
    
    return f

def truncated_svd(A, k):
    U, S, Vh = np.linalg.svd(A, full_matrices=False)
    U_k = U[:, :k]
    S_k = np.diag(S[:k])
    Vh_k = Vh[:k, :]
    return U_k, S_k, Vh_k

def FET(Gamma, sigma, chi_t, max_iter = 20, verbose = 1):

    if verbose:
        print("FET target rank {0} | max_iter {1}".format(chi_t, max_iter))

    # normalize \psi
    psi_psi = ncon([sigma, Gamma, sigma], ((1, 2), (1, 2, 3, 4), (3, 4)))
    Gamma = Gamma/psi_psi
    psi_psi = ncon([sigma, Gamma, sigma], ((1, 2), (1, 2, 3, 4), (3, 4)))

    u, s, v = truncated_svd(sigma, chi_t)
    
    F = [fidelity(Gamma, sigma, u@s@v)]
    U, S, V = [u], [s], [v]

    if verbose:
        print("initial fidelity is {0}".format(F[-1]))

    # iterative optimization of u,s,v = svd(sigma_t)
    uvcir = itt.cycle(["u","v"])
    for k in range(max_iter):
        fix = next(uvcir)
        if fix == "u":
           # R = s @ v
            P = ncon([sigma, Gamma, u], ((1, 2), (1, 2, 3, -2), (3, -1)))
            B = ncon([u, Gamma, u], ((1, -1), (1, -2, 2, -4), (2, -3)))

            shp = B.shape
            B = B.reshape(np.prod(shp[:2]), np.prod(shp[2:]))
            Rmax = P.flatten() @ np.linalg.pinv(B, rcond=1e-10, hermitian=True)
            Rmax = Rmax.reshape(shp[:2])
            u, s, v = truncated_svd(u@Rmax, chi_t)

        elif fix == "v":
           # L = u @ s
            P = ncon([sigma, Gamma, v], ((1, 2), (1, 2, -1, 3), (-2, 3)))
            B = ncon([v, Gamma, v], ((-2, 1), (-1, 1, -3, 2), (-4, 2)))

            sz = B.shape
            B = B.reshape(np.prod(sz[:2]), np.prod(sz[2:]))
            Lmax = P.flatten() @ np.linalg.pinv(B, rcond=1e-10, hermitian=True)
            Lmax = Lmax.reshape(sz[:2])
            u, s, v = truncated_svd(Lmax@v, chi_t)
        
        # s = scale_fix(s, sigma, Gamma, u@s@v)
        F.append(fidelity(Gamma, sigma, u@s@v))

        if verbose:
            print("iteration {0} fideltity {1}".format(k, F[-1]))
    return U, S, V, F

if __name__ == "__main__":
    chi, chi_t = 8, 1
    d = 16
    # A, B, C, D = np.random.rand(chi, chi, chi, chi),  np.random.rand(chi, chi, chi, chi),  np.random.rand(chi, chi, chi, chi),  np.random.rand(chi, chi, chi, chi)
    # Gamma = ncon([A,B,C,D,A,B,C,D], ((1, 2, -1, 3), (4, 5, 6, 2), (-2, 7, 8, 9), (6, 10, 11, 7), 
    #                     (1, 12, -3, 3), (4, 5, 13, 12), (-4, 14, 8, 9), (13, 10, 11, 14)))

    A = np.random.rand(1, chi, chi, 1, d)
    B = np.random.rand(chi, chi, 1, 1, d)
    C = np.random.rand(1, 1, chi, chi, d)
    D = np.random.rand(chi, 1, 1, chi, d)
    A, B, C, D = A/np.linalg.norm(A), B/np.linalg.norm(B), C/np.linalg.norm(C), D/np.linalg.norm(D)
    T = ncon([A,B,C,D], ((-5, 1, 3, -1, -6), (3, 4, -7, -8, -2), (-9, -10, 2, 1, -3), (2, -11, -12, 4, -4)))
    T = np.squeeze(T)
    print("shape of tensor T is {0}".format(T.shape))

    Gamma = ncon([A,B,C,D,A,B,C,D], ((-5, 2, -1, -6, 1), (-2, 4, -7, -8, 9), (-9, -10, 3, 2, 10), (3, -11, -12, 4, 8), 
                                     (-13, 5, -3, -14, 1), (-4, 7, -15, -16, 9), (-17, -18, 6, 5, 10), (6, -19, -20, 7, 8)))
    Gamma = np.squeeze(Gamma)
    print("shape of environment tensor Gamma is {0}".format(Gamma.shape))
    nrm = ncon([Gamma, Gamma], ((1, 2, 3, 4), (1, 2, 3, 4)))
    print('norm of environment tensor Gamma: {0}'.format(nrm))

    sigma = np.eye(chi)
    U, S, V, F = FET(Gamma, sigma, chi_t, max_iter=5)

    plt.semilogy(F)
    plt.show()