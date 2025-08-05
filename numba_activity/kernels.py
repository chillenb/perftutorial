from numba import njit, prange
import numpy as np

from scipy.spatial import distance_matrix

def ref_total_energy(coords):
    dist_matrix = distance_matrix(coords.T, coords.T)
    U_sum = np.sum(1 / (1 + dist_matrix))
    U_sum -= np.sum(np.diag(1 / (1 + dist_matrix)))  # remove self-interaction
    return U_sum / 2  # divide by 2 to account for double counting


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
                U = 1 / (1.0 + np.sqrt(r2))
                tot_e += U
    return tot_e/2
