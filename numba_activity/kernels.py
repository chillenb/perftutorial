from numba import njit, prange
import numpy as np

@njit(cache=True)
def total_energy(coords):
    # sum_{i<j} U(r_i, r_j)
    tot_e = 0.0
    n = coords.shape[1]
    for i in range(n):
        for j in range(n):
            if i != j:
                dx = coords[0, i] - coords[0, j]
                dy = coords[1, i] - coords[1, j]
                dz = coords[2, i] - coords[2, j]
                r2 = dx * dx + dy * dy + dz * dz
                U = np.exp(-np.sqrt(r2))
                tot_e += U
    return tot_e/2
