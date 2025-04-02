import numpy as np
import math
from ncon import ncon
import itertools as itt
from matplotlib import pyplot as plt
import pickle

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
        print("------ F.E.T. target rank {0}, max_iter {1} ------ ".format(chi_t, max_iter))

    # normalize \psi
    psi_psi = ncon([sigma, Gamma, sigma], ((1, 2), (1, 2, 3, 4), (3, 4)))
    Gamma = Gamma/psi_psi
    psi_psi = ncon([sigma, Gamma, sigma], ((1, 2), (1, 2, 3, 4), (3, 4)))

    u, s, v = truncated_svd(sigma, chi_t)
    
    F = [fidelity(Gamma, sigma, u@s@v)]
    U, S, V = [u], [s], [v]

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

        U.append(u)
        S.append(s)
        V.append(v)
        F.append(fidelity(Gamma, sigma, u@s@v))

        if verbose:
            print("  iteration {0} fideltity {1}".format(k, F[-1]))
    print("----------------------------------------------\n")
    return U, S, V, F

def test_fet():
    chi, chi_t = 2, 1 # starting, truncated bond dimension
    d = 2 # physical dimension

    # tensors in a 2x2 peps
    A = np.random.rand(1, chi, chi, 1, d)
    B = np.random.rand(chi, chi, 1, 1, d)
    C = np.random.rand(1, 1, chi, chi, d)
    D = np.random.rand(chi, 1, 1, chi, d)

    # normalize each tensor
    A, B, C, D = A/np.linalg.norm(A), B/np.linalg.norm(B), C/np.linalg.norm(C), D/np.linalg.norm(D)
    
    # full tensor
    T = ncon([A,B,C,D], ((-5, 1, 3, -1, -6), (3, 4, -7, -8, -2), (-9, -10, 2, 1, -3), (2, -11, -12, 4, -4)))
    T = np.squeeze(T)
    print("shape of tensor T is {0}".format(T.shape))

    # SVD on 1st vertical bond
    theta = ncon([A,B], ((-1, -2, 1, -3, -4), (1, -5, -6, -7, -8)))
    shp = theta.shape
    theta = np.reshape(theta, (np.prod(shp[:4]), np.prod(shp[4:])))
    u, s, v = truncated_svd(theta, chi)

    At1 = np.reshape(u[:,:chi_t]@s[:chi_t,:chi_t], (*shp[:4], u[:,:chi_t].shape[1]))
    At1 = np.transpose(At1, (0, 1, 4, 2, 3))
    Bt1 = np.reshape(v[:chi_t,:], (v[:chi_t,:].shape[0], *shp[4:]))

    Tt1 = ncon([At1,Bt1,C,D], ((-5, 1, 3, -1, -6), (3, 4, -7, -8, -2), (-9, -10, 2, 1, -3), (2, -11, -12, 4, -4)))
    Tt1 = np.squeeze(Tt1)
    naive_t_err = np.linalg.norm(T-Tt1)/np.linalg.norm(T)

    # setup for FET
    # initialize with s as bond matrix:
    # A = np.reshape(u, (*shp[:4], u.shape[1]))
    # A = np.transpose(A, (0, 1, 4, 2, 3))
    # B = np.reshape(v, (v.shape[0], *shp[4:]))
    # sigma = s

    # or not: 
    sigma = np.eye(chi)

    Gamma = ncon([A,B,C,D,A,B,C,D], ((-5, 2, -1, -6, 1), (-2, 4, -7, -8, 9), (-9, -10, 3, 2, 10), (3, -11, -12, 4, 8), 
                                     (-13, 5, -3, -14, 1), (-4, 7, -15, -16, 9), (-17, -18, 6, 5, 10), (6, -19, -20, 7, 8)))
    
    Gamma = np.squeeze(Gamma)
    U, S, V, F = FET(Gamma, sigma, chi_t, max_iter=5)
    fet_err = []
    for i in range(len(S)):
        u, s, v = U[i], S[i], V[i]
        At2 = ncon([u@s, A], ((1, -3), (-1, -2, 1, -4, -5)))
        Bt2 = ncon([v, B], ((-1, 1), (1, -2, -3, -4, -5)))

        Tt2 = ncon([At2,Bt2,C,D], ((-5, 1, 3, -1, -6), (3, 4, -7, -8, -2), (-9, -10, 2, 1, -3), (2, -11, -12, 4, -4)))
        Tt2 = np.squeeze(Tt2)
        fet_err.append(np.linalg.norm(T-Tt2)/np.linalg.norm(T))


    # print
    print('SVD rel. trunc. error: {0}'.format(naive_t_err))
    print('FET rel. trunc. error: {0}'.format(fet_err[-1]))

    # plot
    fig, axs = plt.subplots(2, 1, figsize=(6, 6))  # 2 rows, 1 column

    axs[0].semilogy(fet_err, label='FET')
    axs[0].semilogy([naive_t_err] * len(S), label='SVD')
    axs[0].legend()  
    axs[0].set_title('relative error')

    axs[1].semilogy(F)
    axs[1].set_title('fidelity')

    plt.xlabel("iteration")
    plt.tight_layout()
    plt.show()


def test_fet_on_iso():
    with open('fet.py tfi_L2_GS.pkl', 'rb') as f:
        peps = pickle.load(f)

    return

if __name__ == "__main__":
    test_fet()