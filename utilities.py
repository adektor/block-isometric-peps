import numpy as np
from ncon import ncon
from scipy.sparse import kron, eye, lil_matrix
from scipy.sparse.linalg import eigsh
import time

def truncated_svd(A, k):
    """
    Truncated SVD

    Arguments
    ---------
    A: matrix to truncate
    k: truncation rank

    Returns
    -------
    U_k: 
    S_k:
    Vh_k:
    err: 
    """
    # start = time.perf_counter()

    U, S, Vh = np.linalg.svd(A, full_matrices=False)
    U_k = U[:, :k]
    S_k = np.diag(S[:k])
    Vh_k = Vh[:k, :]
    err = np.linalg.norm(S[k+1:])

    # end = time.perf_counter()
    # print("Time to svd ({0},{1}): {2} seconds".format(A.shape[0], A.shape[1], end - start))

    return U_k, S_k, Vh_k, err

def rotate_core(c):
    """
    Rotate a block PEPS core C counter clockwise 

    Arguments
    ---------
    c: 6d numpy array

    Returns
    -------
    Rotated c
    """
    if c.ndim == 6:
        return c.transpose([1, 2, 3, 0, 4, 5])
    else:
        raise ValueError("tensor core has incorrect dimensions")

def full_TFI_matrix_2D(Lx, Ly, J = 1.0, g = 3.5):
    """
    Construct full Hamiltonian for a 2D transverse field Ising model
    on an Lx x Ly lattice. 

    Arguments
    ---------
      Lx - number of sites in the horizontal direction 
      Ly - number of sites in the vertical direction
      J  - coupling strength
      g  - transverse field strength 

    Output:
      H  - Sparse matrix representation of the Hamiltonian
    """
    # total number of sites
    N = Lx * Ly

    # Pauli matrices
    sx = np.array([[0, 1], [1, 0]])  
    sz = np.array([[1, 0], [0, -1]]) 
    id_matrix = eye(2, format='lil')
    
    # Initialize sparse Hamiltonian
    H = lil_matrix((2**N, 2**N))

    # helper function to apply a local operator
    def local_operator(op_local, site):
        """Constructs a local operator acting on a specific site."""
        ops = [id_matrix] * N
        ops[site] = op_local
        op = 1
        for k in range(N):
            op = kron(op, ops[k], format='lil')
        return op

    # nearest-neighbor interactions (-J * sigma_z * sigma_z)
    for x in range(Lx):
        for y in range(Ly):
            site = (y) * Lx + x  # Current site index

            # Right neighbor (x + 1)
            if x < Lx - 1:
                neighbor = site + 1
                H -= J * (local_operator(sz, site) @ local_operator(sz, neighbor))

            # Down neighbor (y + 1)
            if y < Ly - 1:
                neighbor = site + Lx
                H -= J * (local_operator(sz, site) @ local_operator(sz, neighbor))

    # transverse field (-g * sigma_x)
    for site in range(N):
        H -= g * local_operator(sx, site)
    return H