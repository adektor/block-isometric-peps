import numpy as np
import scipy.linalg as la
from ncon import ncon
import copy
from mps import * 
from utilities import *
import time
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
index 4 is physical dimension, index 5 corresponds to block. 
The PEPS is stored as a list of lists of these tensors the block is always 
contained in the orthogonality center. 
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

        if self.Lx == 2 and self.Ly == 2:
            peps_flat = [item for sublist in self.peps for item in sublist]
            v = ncon(peps_flat, ((-9, 1, 2, -10, -1, -5), 
                                 (2, 4, -15, -16, -2, -7), 
                                 (-11, -12, 3, 1, -3, -6), 
                                 (3, -13, -14, 4, -4, -8)))

        elif self.Lx == 1 and self.Ly == 3:
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

    def tebd2(self, Os, dt, Nsteps = None, min_dE = None):
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
            info: dict containing "exp_vals", "tebd_err", "mm_err"

            Modifies
            --------
            self.peps
        """

        Us = [time_evol(Os[0], dt), time_evol(Os[1], dt)]
        Us2 = [time_evol(Os[0], dt/2), time_evol(Os[1], dt/2)]
        d, L = Os[0][0].shape[0], len(Os[0])+1
        Id = [np.reshape(np.eye(d**2), [d] * 4)] * (L-1)

        if min_dE is None:
            min_dE = float("inf")
        if Nsteps is None:
            Nsteps = float("inf")

        p = self.peps[0][0].shape[-1]

        info = dict(exp_vals = np.zeros((p, Nsteps+1)), 
                    tebd_err = np.zeros((Nsteps+1)), 
                    mm_err = np.zeros((Nsteps+1)))

        step = 0
        while step < Nsteps:
            if step % 1 == 0:
                print("iteration {0} out of {1} with max bond dim {2}".format(step, Nsteps, self.tp["tebd_params"]["chi_max"]))
            info_ = self._sweep_and_rotate_4x([Us2[0], Us[1], Us2[0], Id],
                                             Os = [None, None, Os[0], Os[1]])
            info["tebd_err"][step] += info_["tebd_err"]
            step += 1

            # compute expectation values at every iteration (slow)
            info_ = self._sweep_and_rotate_4x([None] * 4,
                                            Os = [None, None, Os[0], Os[1]])
            
            info["exp_vals"][:,step] += info_["exp_vals"]
            info["mm_err"][step] += info_["mm_err"]

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
            tebd_err = 0.0,
            mm_err = 0.0
            )

        for i in range(4):
            info_ = self._sweep_over_cols(Us[i], Os[i])

            info["tebd_err"] += np.sum(info_["tebd_err"])
            info["mm_err"] += np.sum(info_["mm_err"])
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
                # if j % 2 == 0:
                    # Q, R, mm_err = b_mm(self.peps[j], self.tp["mm_params"], dir='up')
                # else:
                    # Q, R, mm_err = b_mm(self.peps[j], self.tp["mm_params"], dir='down')
        
                Q, R, mm_err = b_mm(self.peps[j], self.tp["mm_params"], dir='down')
                info["mm_err"].append(mm_err)
                self.peps[j] = Q

                self.peps[j+1] = pass_R(R, self.peps[j+1])
                self.peps[j+1] = orthogonalize(self.peps[j+1], 'down')
                self.peps[j+1] = truncate(self.peps[j+1], self.tp["tebd_params"]["chi_max"])
    
            else:
                # self.peps[j] = orthogonalize(self.peps[j], 'down')
                self.peps[j] = orthogonalize(self.peps[j], 'up')

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
        rpeps = [[None] * Lx for i in range(Ly)] # initialize rotated peps
        for x in range(Lx):
            for y in range(Ly):
                rpeps[y][x] = rotate_core(peps[Lx - x - 1][y]).copy()

        self.peps = rpeps
        self.Lx = Ly
        self.Ly = Lx

    
    def print(self):
        L = self.Lx
        print("\n")
        for j in range(L):
            # horizontal bond dimensions
            print("\t" + "   ".join(f"[-----]  {self.peps[i][j].shape[1]}" for i in range(L-1)) + "   [-----]")

            # Print horizontal connections
            if j < L-1:
                print("\t" + "|     |------" * (L-1) + "| " + "    |")
                print("\t" + "      ".join("[-----]" for _ in range(L-1)) + "      [-----]")
                print("\t" + "   |         " * (L-1) + "   | ")

            # vertical bond dimensions
            if j < L-1:
                print("\t" + "   | " + "          | ".join(str(self.peps[i][j].shape[2]) for i in range(L-1)) + "          | " + str(self.peps[j][-1].shape[2]))
                print("\t" + "   |         " * (L-1) + "   | ")

        # last row
        print("\t" + "|     |------" * (L-1) + "| " + "    |")
        print("\t" + "      ".join("[-----]" for _ in range(L-1)) + "      [-----]")
        print("\n")

def disentangle(B, nsl, nsr, nb, nc, Niters):
    """ Disentangler computed using alternating optimization 
        
        Arguments
        ---------
        B: (nsl*nsr) x (nb*nc) matrix
        Niters: number of disentangling iterations

        Returns
        -------
        Q: (nsl*nsr) x (nsl*nsr) disentangler for B
    """

    def A(mat, l, r, b, c):
        # reshapes and permutes a matrix of dimension lr x bc to lc x rb
        t = np.reshape(mat, (l, r, b, c))
        tp = np.transpose(t, (0, 3, 1, 2))
        return np.reshape(tp, (l*c, r*b))
    
    def Ainv(mat, l, r, b, c):
        #reshapes and permutes a matrix of dimension lc x rb to lr x bc
        t = np.reshape(mat, (l, c, r, b))
        tp = np.transpose(t, (0, 2, 3, 1))
        return np.reshape(tp, (l*r, b*c))

    Q = [np.eye(nsl * nsr)]
    Amat = A(Q[-1]@B, nsl, nsr, nb, nc)
    u, s, v, err_old = truncated_svd(Amat, 1)
    Amatr = u@s@v
    M = Ainv(Amatr, nsl, nsr, nb, nc)@(B.T)
    u, _, v = np.linalg.svd(M, full_matrices=False)
    Q.append(u@v)

    for i in range(Niters):
        Amat = A(Q[-1]@B, nsl, nsr, nb, nc)
        u, s, v, err_new = truncated_svd(Amat, 1)
        Amatr = u@s@v
        M = Ainv(Amatr, nsl, nsr, nb, nc)@(B.T)
        u, _, v = np.linalg.svd(M, full_matrices=False)
        Q.append(u@v)

        dE = abs(err_old - err_new)
        err_old = err_new

        if dE < 1e-8:
            break
        
        # if i % 10 == 0:
        #     print('disentangler iteration {0}, truncation error {1}'.format(i, err_new))
        
    return Q[-1]

def b_mm(X, mm_params, dir='down'):
    """
    Sequential Moses move for block PEPS.

    Arguments
    ---------
        X: PEPS column
        "mm_params": {"chiV_max": chi, "chiH_max": chi, "etaV_max": chi, "etaH_max": chi, 
                      "n_dis_iters": Niters}
        dir: direction of sweep (and result vertical isometry arrows)

    Returns
    -------
        Q: PEPS column of isometric tensor cores.
        R: PEPS column with no physical indices
        err: accumulated truncation error from all SVDs 
            (does NOT directly indicate the global PEPS error of mm)
    """

    chiV_max, chiH_max, etaV_max, etaH_max = mm_params["chiV_max"], mm_params["chiH_max"], mm_params["etaV_max"], mm_params["etaH_max"]

    if dir == 'up':
        X = flip_col(X)
        flipped_col = True
    else:
        flipped_col = False

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
            D = disentangle(ThetaMatrix1, nsl, nsr, nb, nc, mm_params["n_dis_iters"])
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

    if flipped_col:
        Q = flip_col(Q)
        R = flip_col(R)

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

def time_evol(O, dt):
    ''' Construct imaginary time evolution operators (Trotter Gates) 
        
        Arguments
        ---------
        O: list of local operators
        dt: time step-size

        Returns
        -------
        Us: list of imaginary time evolution operators
    '''
    Us = []
    d = O[0].shape[0] # local Hilbert space dimension
    for H in O:
        H = H.reshape([d*d, d*d])
        U = la.expm(-dt * H).reshape([d] * 4)
        Us.append(U)
    return Us