import numpy as np
from ncon import ncon
from utilities import *

"""
    MPS methods for PEPS column
    
    Methods
    -------
    tebd: Time evolving block decimation sweep from top-to-bottom on a PEPS column. 
          Shifts orthogonality center and block core from top-to-bottom. 

    orthogonalize: Orthogonalizes a PEPS column from bottom-to-top.
"""

def tebd(C, U, O, tebd_params):
    """ Time evolving block decimation on a block isometric PEPS column. 

        Arguments
        ---------
        C: PEPS column (list of numpy arrays)
        U: lists of 2-site Trotter gates
        O: lists of 2-site Hamiltonian terms
        tebd_params: dict of truncation parameters 
                     with keys "chi_max", "svd_tol"

        Returns
        -------
        C: updated PEPS column
        info: dict w/ keys "exp_vals", "tebd_err"
              note that "tebd_err" does directly indicate the global PEPS error
    """
    
    p = C[0].shape[-1] # block size
    L = len(C)         # column length
    c = C[0]           # orthogonality center

    C[0] = orth_block(C[0])
    exp_vals = np.zeros((p,))
    tebd_err = 0.0
    for j in range(L-1):
    
        # two-site block
        theta = ncon([c, C[j+1]], ((-1, -2, 1, -3, -4, -9), (1, -5, -6, -7, -8, -10)))

        # time evolution
        if U != [None]:
            theta = ncon([U[j], theta], ((-4, -8, 1, 2), (-1, -2, -3, 1, -5, -6, -7, 2, -9, -10)))

        # expectation values
        if O != [None]:
            e = ncon([theta, O[j], theta], ((3, 4, 5, 1, 6, 8, 7, 2, -1, 11), (1, 2, 9, 10), (3, 4, 5, 9, 6, 8, 7, 10, -2, 11)))
            exp_vals += np.diag(e)

        # truncate
        shp = theta.shape
        theta = np.reshape(theta, (np.prod(shp[:4]), np.prod(shp[4:9])))
        A, S, B, err = truncated_svd(theta, tebd_params["chi_max"])
        SB = S@B
        tebd_err += err
        A = np.reshape(A, (*shp[:4], A.shape[1]))
        A = np.transpose(A, (0, 1, 4, 2, 3))
        A = np.expand_dims(A, axis=-1)

        SB = np.reshape(SB, (SB.shape[0], *shp[4:9]))
        c = SB

        C[j] = A

    C[L-1] = c
    info = dict(exp_vals = exp_vals,
                tebd_err = tebd_err,
                )
    
    return C, info

def orthogonalize(C, chi = 1000):
    """ Orthogonalize a PEPS column from bottom-to-top 

        Arguments
        ---------
        C: PEPS column

        Returns
        -------
        C: PEPS column with orthogonality center at top site
    """

    L = len(C)

    for i in range(L-1, 0, -1):
        c = C[i]
        shp = c.shape
        c = np.transpose(c, (0, 5, 1, 2, 3, 4))
        c = np.reshape(c, (shp[0]*shp[5], np.prod(shp[1:5])))
        u, s, v, err = truncated_svd(c, chi)
        us = u@s
        C[i] = np.reshape(v, (v.shape[0], *shp[1:5], 1))
        us = np.reshape(us, (shp[0], shp[5], u.shape[1]))
        C[i-1] = ncon([us, C[i-1]], ((1, -6, -3), (-1, -2, 1, -4, -5, -7)))
        C[i-1] = np.squeeze(C[i-1], axis=-1)
    return C

def orth_block(c): 
    """ Gram-Schmidt on block core. Assumes block is upper left (0,0) site """
    p = c.shape[-1]
    for i in range(p):
        ci = c[:,:,:,:,:,i]
        ci = ci / np.linalg.norm(ci)

        for j in range(i + 1, p):
            cj = c[:,:,:,:,:,j]
            cj = cj/np.linalg.norm(cj)
            ci = ci - (np.dot(cj.flatten(), ci.flatten())) * cj

        ci = ci / np.linalg.norm(ci)
        c[:,:,:,:,:,i] = ci
    return c