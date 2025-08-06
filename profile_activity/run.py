import line_profiler
from pyscf import lib
import numpy as np

einsum = lib.einsum

from ref import get_sigma_diag_minimal_ref


@line_profiler.profile
def get_sigma_diag_minimal():
    nocc = 30
    nvir = 150
    naux = 500

    nmo = nocc + nvir

    orbs = range(20, 60)
    norbs = len(orbs)

    nw = nw_sigma = 10
    freqs = np.linspace(0.01, 0.1, nw_sigma)
    wts = np.ones((nw_sigma))
    rng = np.random.default_rng(0)

    mo_energy = rng.random((nmo))
    Lpq = rng.random((naux, nmo, nmo))
    lib.hermi_sum(Lpq, axes=(0, 2, 1), inplace=True)

    ef = 0

    omega = np.zeros((nw_sigma))
    omega[1:] = freqs[: (nw_sigma - 1)]
    emo = omega[None] + ef - mo_energy[:, None]
    sigma = np.zeros((norbs, nw_sigma))
    for w in range(nw):
        # Pi_inv = 1 - (1 - Pi)^{-1} - 1 = -[1 + (Pi - 1)^{-1}]
        Pi = get_rho_response(freqs[w], mo_energy, Lpq[:, :nocc, nocc:])
        Pi[range(naux), range(naux)] -= 1.0
        Pi_inv = np.linalg.inv(Pi)
        Pi_inv[range(naux), range(naux)] += 1.0
        Qnm = einsum("Pnm, PQ -> Qnm", Lpq[:, orbs], Pi_inv)
        Wmn = einsum("Qnm, Qmn -> mn", Qnm, Lpq[:, :, orbs])
        g0 = wts[w] * emo / (emo**2 + freqs[w] ** 2)
        sigma += einsum("mn, mw -> nw", Wmn, g0) / np.pi
    return sigma


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
    Pi = einsum("Pia, Qia -> PQ", Pia, Lpq)
    return Pi


if __name__ == "__main__":
    sol = get_sigma_diag_minimal_ref()
    sol2 = get_sigma_diag_minimal()
    if not np.allclose(sol, sol2, rtol=1e-6):
        raise ValueError(f"Results do not match: {sol} != {sol2}")
    for _ in range(2):
        get_sigma_diag_minimal()
