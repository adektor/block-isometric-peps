from block_iso_peps import *
import matplotlib.pyplot as plt
import pickle
# import pdb

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

def iso_tebd_ising_2D(L, J, g, p, dts, Nt, t_params):
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

    Os = [TFI_bonds(L, J, g), TFI_bonds(L, J, 0)]                   # [vertical bonds, horizontal bonds]
    peps = b_iso_peps(random_peps(L, L, 2, p), t_params)

    info = dict(exp_vals = [], 
                    tebd_err = [],
                    mm_err = [])
    
    for dt in dts:
        print(("\nTEBD2 with dt = {0}\n" + "-" * 15).format(dt))
        info_ = peps.tebd2(Os, dt, Nsteps = Nt)
        
        info["exp_vals"].append(info_["exp_vals"][:,1:])
        info["tebd_err"].append(info_["tebd_err"])
        info["mm_err"].append(info_["mm_err"])
        
    info["exp_vals"] = np.hstack(info["exp_vals"])
    info["tebd_err"] = np.hstack(info["tebd_err"])
    info["mm_err"] = np.hstack(info["mm_err"])
    print("Done")

    return peps, info

if __name__ == '__main__':
    L, Nt = 3, 300
    J, g = 1, 3.5
    p = 1
    dts = [0.01]
    chi = 16
    t_params = {"tebd_params": {"chi_max": chi, "svd_tol": 0}, 
                "mm_params": {"chiV_max": chi, "chiH_max": chi, "etaV_max": chi, "etaH_max": chi, "n_dis_iters": 100}}
    
    peps, info = iso_tebd_ising_2D(L, J, g, p, dts, Nt, t_params)
    peps.print()

    # save PEPS
    # with open('tfi_L2_GS.pkl', 'wb') as f:
    #     pickle.dump(peps, f)
    
    # get reference eigenvalues
    H = full_TFI_matrix_2D(L, L, J, g)
    E_ref, _ = eigsh(H, k=p, which='SA')
    E_ref = np.expand_dims(E_ref, axis=1)

    # sort PEPS eigenvalues
    E = np.sort(info["exp_vals"][:,-1])

    # print reference and PEPS eigenvalues
    for i in range(p):
        print(f"ref. eig {i}: {E_ref[i][0]}")
        print(f"peps eig {i}: {E[i]} \n")

    exp_vals = np.sort(info["exp_vals"], axis=0)
    en_den_err = np.abs(exp_vals - E_ref)/(L**2)

    # plot
    plt.figure(1) # energy density errors
    for i in range(en_den_err.shape[0]):
        plt.semilogy(en_den_err[i,:], label="eigenvalue {0}".format(i))
    plt.xlabel("iteration")  
    plt.ylabel("energy density error")      
    plt.legend()
    

    plt.figure(2) # TEBD, moses move errors
    plt.semilogy(info["tebd_err"], label="tebd")
    plt.semilogy(info["mm_err"], label="moses move")
    plt.xlabel("iteration")  
    plt.ylabel("truncation error")      
    plt.legend()

    plt.show()