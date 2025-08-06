import line_profiler
from pyscf import lib
import numpy as np
einsum = lib.einsum


@line_profiler.profile
def get_sigma_diag_minimal():
    nocc = 30
    nvir = 150
    naux = 300
    norbs = nocc + nvir
    nw = nw_sigma = 10
    freqs = np.linspace(0.01, 0.1, nw_sigma)
    wts = np.ones((nw_sigma))

    mo_energy = np.random.random((norbs))
    Lpq = np.random.random((naux, norbs, norbs))

    ef = 0

    omega = np.zeros((nw_sigma), dtype=np.complex128)
    omega[1:] = 1j*freqs[:(nw_sigma-1)]
    emo = omega[None] + ef - mo_energy[:, None]
    sigma = np.zeros((norbs, nw_sigma), dtype=np.complex128)
    for w in range(nw):
        # Pi_inv = 1 - (1 - Pi)^{-1} - 1 = -[1 + (Pi - 1)^{-1}]
        Pi = get_rho_response(freqs[w], mo_energy, Lpq[:, :nocc, nocc:])
        Pi[range(naux), range(naux)] -= 1.0
        Pi_inv = np.linalg.inv(Pi)
        Pi_inv[range(naux), range(naux)] += 1.0
        Qnm = einsum('Pnm, PQ -> Qnm', Lpq, Pi_inv)
        Wmn = einsum('Qnm, Qmn -> mn', Qnm, Lpq)
        g0 = wts[w] * emo / (emo**2 + freqs[w]**2)
        sigma += einsum('mn, mw -> nw', Wmn, g0) / np.pi


@line_profiler.profile
def get_rho_response(omega, mo_energy, Lpq):
    """
    Compute density response function in auxiliary basis at freq iw.
    """
    naux, nocc, nvir = Lpq.shape
    eia = mo_energy[:nocc, None] - mo_energy[None, nocc:]
    eia = eia / (omega**2 + eia * eia)
    # Response from both spin-up and spin-down density
    Pia = Lpq * (eia * 4.0)
    Pi = einsum('Pia, Qia -> PQ', Pia, Lpq)
    return Pi


if __name__ == '__main__':
    get_sigma_diag_minimal()
