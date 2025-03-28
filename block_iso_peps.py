import numpy as np
from ncon import ncon
import copy
from mps import * 
from utilities import *
# from misc import *
# from tebd import tebd, get_time_evol

"""
AD March 2025. Block isometric tensor networks states in 2D (PEPS).
PEPS tensor index convention:
         0  5                    
         | /                        
     3---T---1
        /|                          
       4 2    
index 4 is physical dimension, index 5 corresponds is block. 
The PEPS is stored as a list of lists of these tensors. 
PEPS[i] returns tensor cores in the ith column. 

This file contains the class BisoPEPS (block isometric PEPS) and some static methods for 
manipulating BisoPEPS, e.g., b_mm (block sequential moses move).
"""

class b_iso_peps:
    """ Block isometric PEPS.

    Attributes
    ----------
    peps: list of lists of numpy arrays
    Lx: horizontal length of lattice
    Ly: vertical length of lattice
    tp: Dictionary of truncation parameters
              Contains subdictionary tebd_params with chi_max and svd_tol, and can optionally 
              contain moses_move truncation_params)
             {tebd_params: {chi_max = chi_max, svd_tol = svd_tol},
              mm_params: {chiV_max, chiH_max, etaV_max, etaH_max}
             }

    Methods
    -------
    The primary method is tebd2 which performs imaginary time evolution 
    on self.peps to find the low-lying eigenvectors of a local Hamiltonian
    """

    def __init__(self, peps, trunc_params):
        self.peps = peps
        self.Lx = len(peps)
        self.Ly = len(peps[0])
        self.tp = trunc_params

    def copy(self):
        return copy.deepcopy(self) 
    
    def contract(self):
        """ Contracts PEPS into full tensor.
            Warning: Only use for small PEPS.
        """

        if self.Lx == 1 and self.Ly == 3:
            v = ncon(self.peps[0], ((-12, -5, 1, -6, -1, -4),
                                    (1, -7, 2, -8, -2, -13), 
                                    (2, -9, -10, -11, -3, -14)))
        elif self.Lx == 3 and self.Ly == 3:
            peps_flat = [item for sublist in self.peps for item in sublist]
            v = ncon(peps_flat,((-11, 1, 2, -12, -1, -10),
                                (2, 6, 7, -16, -2, -25), 
                                (7, 11, -18, -19, -3, -28),
                                (-13, 3, 4, 1, -4, -23), 
                                (4, 8, 9, 6, -5, -26), 
                                (9, 12, -20, 11, -6, -29), 
                                (-14, -15, 5, 3, -7, -24), 
                                (5, -17, 10, 8, -8, -27), 
                                (10, -21, -22, 12, -9, -30)))
        else:
            v = 0
            print("PEPS dimensions not supported for contraction")
        return np.squeeze(v)

    def tebd2(self, Os, Us, Nsteps = None, min_dE = None):
        """ Time evolving block decimation on isometric PEPS (TEBD^2)
            Applies time evolution gates to rows and columns of PEPS 
            while periodically orthogonalizing the block core. 

            Arguments
            ---------
            Os: list containing lists of vertical and horizontal 2-site Hamiltonians
                for computing expectation values
            Us: list containing lists of vertical and horizontal 2-site time evolution operators
            Nsteps: number of time steps
            min_dE: break after the change in energy between sweeps is less than min_dE

            Returns
            -------
            info: Dict on final run 

            Modifies
            --------
            self.peps
        """

        if min_dE is None:
            min_dE = float("inf")
        if Nsteps is None:
            Nsteps = float("inf")

        p = self.peps[0][0].shape[-1]
        info = dict(exp_vals = np.zeros((p, Nsteps+1)))

        step = 0
        while step < Nsteps:
            if step % 50 == 0:
                print("iteration {0} out of {1} with max bond dim {2}".format(step, Nsteps, self.tp["tebd_params"]["chi_max"]))
            _ = self._sweep_and_rotate_4x([Us[0], Us[1], Us[0], Us[1]],
                                             Os = [None, None, Os[0], Os[1]])
            
            step += 1

            # compute expectation values at every iteration (slow)
            info_ = self._sweep_and_rotate_4x([None] * 4,
                                            Os = [None, None, Os[0], Os[1]])
            
            info["exp_vals"][:,step] = info_["exp_vals"]

        return info
    
    def _sweep_and_rotate_4x(self, Us, Os = None):
        """ Sweep over all columns performing TEBD and then rotate 4 times 
            to perform TEBD on all columns and rows twice.
        
            Arguments
            ---------
            Us: List of 4 (one for each sweep) lists of Trotter gates
            Os: List of 4 (one for each sweep) lists of 2-site Hamiltonian terms
            
            Returns
            -------
            info: Dictionary of information about the peps.

            Modifies
            --------
            self.peps
        """
        
        if Us is None:
            Us = [None] * 4
        if Os is None:
            Os = [None] * 4

        p = self.peps[0][0].shape[-1]
        info = dict(exp_vals = np.zeros(p,),
            tebd_err = [0.0] * 4,
            moses_err = [0.0] * 4
            )

        for i in range(4):
            # self.orth_block()
            # print("Starting sequence {i} of full sweep".format(i=i))
            info_ = self._sweep_over_cols(Us[i], Os[i])

            info["tebd_err"][i] += np.sum(info_["tebd_err"])
            info["moses_err"][i] += np.sum(info_["mm_err"])
            if i > 1:
                info["exp_vals"] += info_["exp_vals"]

            self._rotate()
        return info

    def _sweep_over_cols(self, U, O = None):
        ''' Perform TEBD on each column of self.peps

            Arguments
            ---------
            U: Lists of 2-site Trotter gates
            O: Lists of 2-site Hamiltonian terms

            Returns
            -------
            info: dict

            Modifies
            --------
            self.peps
        '''

        info = dict(exp_vals = [],
                    tebd_err = [],
                    mm_err = [],
                    nrm = 1.0)
        
        if U is None:
            U = [None]
        if O is None:
            O = [None]

        Lx, Ly = self.Lx, self.Ly

        p = self.peps[0][0].shape[-1]
        exp_vals = np.zeros(p,)

        for j in range(Lx):
            self.peps[j], tebd_info = tebd(self.peps[j], U, O, self.tp["tebd_params"])
            exp_vals += tebd_info["exp_vals"]
            info["tebd_err"].append(tebd_info["tebd_err"])

            if j < Lx - 1:
                Q, R, mm_err = b_mm(self.peps[j], self.tp["mm_params"])
                info["mm_err"].append(mm_err)
                self.peps[j] = Q
                self.peps[j+1] = pass_R(R, self.peps[j+1])

                # may need to insert a truncation here...
            else:
                self.peps[j] = orthogonalize(self.peps[j])

        info["exp_vals"] = exp_vals

        return info
    

    def _rotate(self):
        """ Rotate self.peps counter-clockwise by 90 degrees

            Modifies
            --------
            self.peps
        """

        peps = self.peps
        Lx, Ly = self.Lx, self.Ly
        rpeps = [[None] * Lx for i in range(Ly)] # rotated dimensions
        for x in range(Lx):
            for y in range(Ly):
                rpeps[y][x] = rotate_core(peps[Lx - x - 1][y]).copy()

        self.peps = rpeps
        self.Lx = Ly
        self.Ly = Lx

    
    def print(self):
        # TODO: print for general Lx, Ly
        # for i in range(self.Ly):
        #     for j in range(self.Lx):
        #         print(f"\t[-----]  {self.peps[0][0].shape[1]} ")
        #         print("\t|     |------ ")
        #         print("\t[-----]     ")
        #         print("\t   |        ")
        #         print(f"\t   | {self.peps[0][2].shape[4]} ")
        #         print("\t   |       ")

        if self.Ly == 3 and self.Lx == 3:
            # DOUBLE CHECK BOND DIMENSIONS ARE CORRECT
            print("\n")
            print(f"\t[-----]  {self.peps[0][0].shape[1]}   [-----]  {self.peps[1][0].shape[1]}   [-----]")
            print("\t|     |------|     |------|     | ")
            print("\t[-----]      [-----]      [-----] ")
            print("\t   |            |            | ")
            print(f"\t   | {self.peps[0][1].shape[0]}          | {self.peps[1][1].shape[0]}          | {self.peps[2][1].shape[0]}")
            print("\t   |            |            |  ")
            print(f"\t[-----]  {self.peps[0][1].shape[1]}   [-----]  {self.peps[0][1].shape[1]}   [-----]")
            print("\t|     |------|     |------|     | ")
            print("\t[-----]      [-----]      [-----] ")
            print("\t   |            |            |        ")
            print(f"\t   | {self.peps[0][2].shape[0]}          | {self.peps[1][2].shape[0]}          | {self.peps[2][2].shape[0]}")
            print("\t   |            |            |        ")
            print(f"\t[-----]  {self.peps[0][2].shape[1]}   [-----]  {self.peps[1][2].shape[1]}   [-----]   ")
            print("\t|     |------|     |------|     | ")
            print("\t[-----]      [-----]      [-----] ")
            print("\n")

def disentangle(matrix, nsl, nsr, nb, nc, dis_options):
    """ Placeholder function for disentangling. Implement as needed. """
    if dis_options.get("type", "none") == "none":
        return np.eye(nsl * nsr)  # Identity if no disentangling is applied
    else:
        raise NotImplementedError("Disentangling method not implemented")

def b_mm(X, mm_params):
    """
    Sequential Moses move for block PEPS.

    Arguments
    ---------
        X: PEPS column
        "mm_params": {"chiV_max": chi, "chiH_max": chi, "etaV_max": chi, "etaH_max": chi, 
                      "disentangle": False}}

    Returns
    -------
        Q: PEPS column of isometric tensor cores.
        R: PEPS column with no physical indices
        err: accumulated truncation error from all SVDs 
            (does NOT directly indicate the global PEPS error of mm)
    """

    # if mm_params["disentangle"] is None:
    dis_options = {"type": "none"} # no disentangler for now...

    chiV_max, chiH_max, etaV_max, etaH_max = mm_params["chiV_max"], mm_params["chiH_max"], mm_params["etaV_max"], mm_params["etaH_max"]

    k = len(X)
    Q = [None] * k
    R = [None] * k
    e1 = np.zeros(k)
    e2 = np.zeros(k)

    # Initialize 6-tensor with bottom core
    sz = X[-1].shape
    C = X[-1].reshape((*sz[:3], 1, *sz[3:]))

    # Sweep upwards
    for i in range(k-1, -1, -1):
        sz = C.shape
        na, nb, nc = np.prod(sz[3:6]), np.prod(sz[1:3]), sz[0] * sz[6]
        C = np.transpose(C, (3, 4, 5, 1, 2, 0, 6))
        U, S, Vh = np.linalg.svd(C.reshape(na, nb * nc), full_matrices=False)
        diagS = S.copy()

        # Truncation
        ns = len(S)
        if i == 0:
            ns2 = min(ns, chiH_max)
        else:
            if ns == 2:
                nsl, nsr = 2, 1
            elif ns == 8:
                nsl, nsr = 4, 2
            elif ns == 32:
                nsl, nsr = 8, 4
            else:
                nsl = min(max(int(np.sqrt(ns)), 1), etaV_max)
                nsr = min(max(int(np.sqrt(ns)), 1), chiH_max)
            ns2 = nsl * nsr

        Ut, St, Vt = U[:, :ns2], np.diag(S[:ns2]), Vh[:ns2, :]
        Theta = St @ Vt
        e1[i] = np.sum(diagS[ns2:] ** 2)

        # Update Q
        if i == 0:
            Q[i] = Ut.reshape((*sz[3:6], 1, ns2))
        else:
            Q[i] = Ut.reshape((*sz[3:6], nsl, nsr))
        Q[i] = np.transpose(Q[i], (3, 4, 0, 1, 2))

        # Second SVD
        if i > 0:
            ThetaTensor = Theta.reshape(nsl, nsr, nb, nc)

            # disentangler
            ThetaMatrix1 = ThetaTensor.reshape(nsl * nsr, nb * nc)
            D = disentangle(ThetaMatrix1, nsl, nsr, nb, nc, dis_options)
            ThetaMatrix1 = D @ ThetaMatrix1
            Dadj = D.T.reshape(nsl, nsr, nsl, nsr)
            Q[i] = ncon([Q[i], Dadj], ((1, 2, -3, -4, -5), (1, 2, -1, -2)))

            ThetaTensor = ThetaMatrix1.reshape(nsl, nsr, nb, nc)
            ThetaTensor = np.transpose(ThetaTensor, (0, 3, 1, 2))
            ThetaMatrix = ThetaTensor.reshape(nsl * nc, nsr * nb)
            UT, ST, VT = np.linalg.svd(ThetaMatrix, full_matrices=False)
            diagST = ST.copy()
            nt = min(len(ST), etaH_max)

            PsiMatrix = UT[:, :nt] @ np.diag(ST[:nt]) / np.linalg.norm(ST[:nt])
            PsiTensor = PsiMatrix.reshape(nsl, sz[0], sz[6], nt)
            e2[i] = np.sum(diagST[nt:] ** 2)

            # Contract with upper core
            C = ncon([X[i - 1], PsiTensor], ((-1, -2, 1, -5, -6, -8), (-4, 1, -7, -3)))
            C = np.squeeze(C,axis=-1)
            R[i] = VT[:nt, :].reshape(nt, nsr, sz[1], sz[2])
            R[i] = np.transpose(R[i], (0, 2, 3, 1))
            R[i] = np.expand_dims(R[i], axis=-1)
        elif i == 0:
            R[i] = Theta.reshape(1, ns2, sz[1], sz[2], sz[6])
            R[i] = np.transpose(R[i], (0, 2, 3, 1, 4))

        Q[i] = np.expand_dims(Q[i], axis=-1)

    err = np.sum(e1) + np.sum(e2)
    return Q, R, err

def pass_R(R, X):
    """ Pass R from mm to right column
    Arguments
    ---------
    R: PEPS column with no physical dimension
    X: PEPS column

    Returns
    -------
    RX: PEPS column
    """
    L = len(X)
    RX = []
    for i in range(L):
        shpX = X[i].shape
        shpR = R[i].shape
        RX.append(ncon([R[i], X[i]], ((-1, 1, -4, -6, -8), (-2, -3, -5, 1, -7, -9))))
        RX[i] = np.reshape(RX[i], (shpR[0]*shpX[0], shpX[1], shpR[2]*shpX[2], shpR[3], shpX[4], shpR[4]*shpX[5]))

    return RX