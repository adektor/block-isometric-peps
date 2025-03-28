import numpy as np
import matplotlib.pyplot as plt
from mps import *
from ising_model_2D import *
from utilities import *

"""
Unit tests for block isometric tns.
Run this file to ensure that all tests pass. 
"""

def test_rotation():
    ''' test rotation of block PEPS cores and columns '''
    # --------- single core rotation -------- #
    T6 = np.random.rand(1, 2, 3, 4, 5, 6)
    rT6 = rotate_core(T6)
    shp, rshp = T6.shape, rT6.shape

    assert shp[0] == rshp[3]
    assert shp[1] == rshp[0]
    assert shp[2] == rshp[1]
    assert shp[3] == rshp[2]
    assert shp[4] == rshp[4]
    assert shp[5] == rshp[5]

    # --------- column rotation -------- #
    Ly, d, p = 3, 2, 1
    peps = b_iso_peps(random_peps(1, Ly, d, p), {})

    v = peps.contract()

    peps._rotate()
    peps._rotate()
    vr = peps.contract()

    assert np.linalg.norm(v - np.transpose(vr, (2, 1, 0))) < 1e-10
    print("- rotation test passed")

def test_matvec():
    ''' test two-site contractions gives the same result as matvec '''
    Lx = 1
    Ly = 3
    d = 2
    p = 1

    J, g = 1, 3.5

    H = full_TFI_matrix_2D(Lx, Ly, J, g) # full Hamiltonian
    O = TFI_bonds(Ly, J, g)              # Hamiltonian as 2-site operators

    peps = b_iso_peps(random_peps(Lx, Ly, d, p), {})

    # --------- reference matvec ---------- #
    v = peps.contract()
    Hv1 = H@v.flatten()

    # -------- apply two-site gates after contracting PEPS --------- #
    Hv2 = ncon([O[0], v], ((-1, -2, 1, 2), (1, 2, -3))) + ncon([O[1], v], ((-2, -3, 1, 2), (-1, 1, 2)))
    err = np.linalg.norm(Hv2.flatten() - Hv1)
    assert err < 1e-10, f"matvec error {err}"

    # -------- apply two-site gates before contracting PEPS --------- #
    # first term
    O1peps = peps.copy() # need to figure out how to copy peps...
    
    theta = ncon([O1peps.peps[0][0], O1peps.peps[0][1]], ((-1, -2, 1, -3, -4, -9), (1, -5, -6, -7, -8, -10)))
    theta = ncon([O[0], theta], ((-4, -8, 1, 2), (-1, -2, -3, 1, -5, -6, -7, 2, -9, -10)))
    shp = theta.shape
    theta = np.reshape(theta, (np.prod(shp[:4]), np.prod(shp[4:9])))
    A, S, B, err = truncated_svd(theta, 32)
    SB = S@B
    A = np.reshape(A, (*shp[:4], A.shape[1]))
    A = np.transpose(A, (0, 1, 4, 2, 3))
    A = np.expand_dims(A, axis=-1)

    SB = np.reshape(SB, (SB.shape[0], *shp[4:9]))

    O1peps.peps[0][0] = A
    O1peps.peps[0][1] = SB

    # second term
    O2peps = peps.copy() # need to figure out how to copy peps...
    
    theta = ncon([O2peps.peps[0][1], O2peps.peps[0][2]], ((-1, -2, 1, -3, -4, -9), (1, -5, -6, -7, -8, -10)))
    theta = ncon([O[1], theta], ((-4, -8, 1, 2), (-1, -2, -3, 1, -5, -6, -7, 2, -9, -10)))
    shp = theta.shape
    theta = np.reshape(theta, (np.prod(shp[:4]), np.prod(shp[4:9])))
    A, S, B, err = truncated_svd(theta, 32)
    SB = S@B
    A = np.reshape(A, (*shp[:4], A.shape[1]))
    A = np.transpose(A, (0, 1, 4, 2, 3))
    A = np.expand_dims(A, axis=-1)

    SB = np.reshape(SB, (SB.shape[0], *shp[4:9]))

    O2peps.peps[0][1] = A
    O2peps.peps[0][2] = SB

    Hv3 = O1peps.contract().flatten() + O2peps.contract().flatten()

    err = np.linalg.norm(Hv3.flatten() - Hv1)
    assert err < 1e-10, f"matvec error {err}"


    print("- matvec test passed")
    return

def test_isoms(peps):
    """
    Check isometries of a PEPS tensor network.
    Ensures PEPS is in a 'canonical' form with center (0,0), i.e., top-left.
    This test is used in:
        - test_col_orth()
        - test_1D_tebd()
    """
    # Assume Lx = 1, i.e., PEPS has a single column

    c = peps.peps[0][0]
    v = peps.contract().flatten()
    assert abs(np.linalg.norm(c.flatten()) - np.linalg.norm(v)) < 1e-10

    c = peps.peps[0][1]
    sz = c.shape
    ov = ncon([c, c], ((-1, 1, 2, -2, 3, -5), (-3, 1, 2, -4, 3, -6)))    
    assert np.linalg.norm(np.reshape(ov, (sz[0]*sz[3], sz[0]*sz[3])) - np.eye(sz[0] * sz[3])) < 1e-10

    c = peps.peps[0][2]
    sz = c.shape
    ov = ncon([c, c], ((-1, 1, 2, -2, 3, -5), (-3, 1, 2, -4, 3, -6)))      
    assert np.linalg.norm(np.reshape(ov, (sz[0]*sz[3], sz[0]*sz[3])) - np.eye(sz[0] * sz[3])) < 1e-10

    return

def test_col_orth():
    Lx, Ly, d, p = 1, 3, 2, 1
    peps = b_iso_peps(random_peps(Lx, Ly, d, p), {})
    peps.peps[0] = orthogonalize(peps.peps[0])
    test_isoms(peps)
    print('- column orth test passed')

def test_1D_block_tebd(plot_error=False):
    """" test 1D TEBD for PEPS column (Lx=1) """
    Lx, Ly, d, p = 1, 3, 2, 3

    peps = b_iso_peps(random_peps(Lx, Ly, d, p), {})
    # peps.peps[0] = orthogonalize(peps.peps[0])

    J, g = 1, 3.5
    dt, Nt = 1e-2, 1000
    H = full_TFI_matrix_2D(Lx, Ly, J, g)
    O = TFI_bonds(Ly, J, g)
    U = time_evol(O, dt)
    tebd_params = {"chi_max": 256, "svd_tol": 0}

    exp_val, tebd_err, nrm = [], [], []
    print("\nRunning block TEBD on 1D Ising model to compute {0} eigenvalues".format(p))
    for i in range(Nt):
        # test_isoms(peps)
        # print('isom test passed in TEBD iteration {0}'.format(i))

        peps.peps[0], info = tebd(peps.peps[0], U, [None], tebd_params)
        tebd_err.append(info["tebd_err"])
        peps._rotate()
        peps._rotate()

        # top core contains block
        peps.orth_block()

        _, info = tebd(peps.copy().peps[0], [None], O, tebd_params)
        exp_val.append(np.sort(info["exp_vals"]))

        if i % 200 == 0:
            print("iteration {0} | lowest eig. {1} | TEBD error {2}".format(i, exp_val[-1][0], tebd_err[-1]))
    
    E_ref = ising_eigs(Lx, Ly, J, g, p)
    print(f"ref  eig 0: {E_ref[0]}")
    print(f"peps eig 0: {exp_val[-1][0]} \n")

    err = np.abs(E_ref - exp_val)

    if plot_error:
        for i in range(err.shape[1]):
            plt.semilogy(err[:,i], label="eigenvalue {0}".format(i))
        plt.xlabel("iteration")  
        plt.ylabel("error")      
        plt.legend()
        plt.show()

    assert err[-1][0] < dt, f"approximate eigenvalue is not close to reference"
    print('- 1D TEBD test passed')

if __name__ == '__main__':
    print(' \n Running unit tests \n' + "-" * 20)
    test_rotation()
    test_matvec()
    test_col_orth()
    test_1D_block_tebd()

    print('All tests passed')