from block_iso_peps import *
import scipy.linalg as la
import pickle
import pdb

"""
    This file is for performing imaginary TEBD 
    to compute excited states of the 2D TFI model 
    using a block isometric tensor network. 
"""

def TFI_bonds(L, J = 1.0, g = 3.5):
    ''' Construct TFI Hamiltonian as a list of 2-site terms
    
        Arguments
        ---------
        L: dimensions of 2D square lattice (L x L)
        J: interaction coefficient
        g: external field coefficient

        Returns
        -------
        O: list of 2-site operators
    '''
    num_bonds = L - 1
    d = 2
    sx = np.array([[0., 1.], [1., 0.]])
    sz = np.array([[1., 0.],[0., -1.]])
    id = np.eye(2)
    O = []
    
    for site in range(num_bonds):
        gL = gR = 0.5 * g
        if site == 0: gL = g
        if site == L - 2: gR = g
        H_local = -J * np.kron(sz, sz) - gL * np.kron(sx, id) - gR * np.kron(id, sx)
        O.append(np.reshape(H_local, [d] * 4))
    return O

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
    d = O[0].shape[0] # Local Hilbert space dimension
    for H in O:
        H = H.reshape([d*d, d*d])
        U = la.expm(-dt * H).reshape([d] * 4)
        Us.append(U)
    return Us

def random_col(L = 3, d = 2, p = 1, block = False):
    ''' Construct a random PEPS column

        Arguments
        ---------
        L: column length
        d: physical dimension
        p: block size
        block: is column block (True/False)

        Returns
        -------
        C: list of PEPS tensor cores
    '''

    C = []
    for i in range(L):
        if i == 0 and block == True:
            c = np.random.rand(1, 1, 1, 1, d, p)
        else:
            c = np.random.rand(1, 1, 1, 1, d, 1)
        C.append(c.copy())
    return C

def random_peps(Lx = 3, Ly = 3, d = 2, p = 1):
    ''' Construct random block PEPS. 
        Top-left tensor core is the block core.

        Arguments
        ---------
        Lx: horizontal length
        Ly: vertical length
        d: physical dimension
        p: block size

        Returns
        -------
        peps: list of lists of tensor cores
    '''

    peps = []
    for i in range(Lx):
        if i == 0:
            peps.append(random_col(Ly, d, p, block = True))
        else:
            peps.append(random_col(Ly, d, p, block = False))
    return peps

def iso_tebd_ising_2D(L, J, g, dts, Nt, t_params):
    ''' Perform block isometric time evolving block decimation on 2D Ising model.

        Arguments
        ---------
        L: dimensions of 2D square lattice (L x L)
        J: interaction coefficient
        g: external field coefficient
        dts: list of time-step sizes
        Nt: # of steps to take with each time step
        t_params: dictionary "chi_max", "svd_tol"

        Returns
        -------
        peps: block isometric PEPS approximating eigenvectors with algebraically smallest eigenvalues
    '''

    Os = [TFI_bonds(L, J, 0), TFI_bonds(L, J, g)]                   # [vertical bonds, horizontal bonds]
    peps = b_iso_peps(random_peps(Lx = L, Ly = L, d = 2), t_params)
    for dt in dts:
        print(("with dt = {0}\n" + "-" * 15).format(dt))
        Us = [time_evol(Os[0], dt), time_evol(Os[1], dt)]
        info = peps.tebd2(Os, Us, Nsteps = Nt, min_dE = 1.e-8)
    print("Done")

    return peps, info

if __name__ == '__main__':
    L, Nt = 3, 100
    J, g = 1, 3.5
    p = 1
    dts = [0.01]
    chi = 100
    t_params = {"tebd_params": {"chi_max": chi, "svd_tol": 0}, 
                "mm_params": {"chiV_max": chi, "chiH_max": chi, "etaV_max": chi, "etaH_max": chi, "disentangle": False}}
    
    peps, info = iso_tebd_ising_2D(L, J, g, dts, Nt, t_params)
    peps.print()

    # check this is correctly computing expectation value ...
    E = info["exp_vals"][-1]

    H = full_TFI_matrix_2D(L, L, J, g)
    E_ref, _ = eigsh(H, k=p, which='SA')

    v = peps.contract()
    print('norm of my vector is {0}'.format(np.linalg.norm(v)))
    print('exp val from full is {0}'.format(-np.linalg.norm(H@v.flatten())))

    print(f"ref  eig 0: {E_ref[0]}")
    print(f"peps eig 0: {E} \n")

    # pdb.set_trace()
    # E = np.sum(info['expectation_O'][2]) + np.sum(info['expectation_O'][3])
    # E_ref = -32.402186096095363
    # E_ref = -130.6117109104326
    # print("Energy density error {0}".format((E-E_ref)/(L**2)))