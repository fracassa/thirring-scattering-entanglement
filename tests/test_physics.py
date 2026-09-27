"""
tests/test_physics.py

Basic physics sanity checks for thirring_sim.py. Written as plain assert-based
functions named `test_*` so they run both with `pytest` (if installed) and
standalone via:

    python tests/test_physics.py
"""

import sys
import os
import numpy as np
from scipy.sparse.linalg import eigsh

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import thirring_sim as ts  # noqa: E402


def test_hamiltonian_is_hermitian():
    for N, m, g in [(8, 0.3, 0.0), (8, 0.4, 0.5), (12, 0.6, 0.9)]:
        H = ts.build_hamiltonian(N, m, g)
        diff = (H - H.getH()).toarray()
        assert np.max(np.abs(diff)) < 1e-10, f"H not Hermitian for N={N}, m={m}, g={g}"
    print("OK: Hamiltonian is Hermitian for all tested parameter sets.")


def test_ground_state_is_normalized():
    H = ts.build_hamiltonian(N=8, m=0.4, g=0.5)
    Omega = ts.ground_state(H)
    norm = np.linalg.norm(Omega)
    assert abs(norm - 1.0) < 1e-8, f"Ground state not normalized: norm={norm}"
    print("OK: ground state is normalized.")


def test_free_theory_vacuum_energy():
    """
    At g=0 the Hamiltonian is exactly solvable in momentum space:
        H = sum_k w_k (c_k^dagger c_k - d_k d_k^dagger)
    which, after normal ordering, gives a vacuum (Dirac sea) energy
        E_vacuum = -sum_k w_k(m)

    IMPORTANT CAVEAT (also present in the paper itself, see README): the
    real-space Hamiltonian (Eq. 9) is built with OPEN boundary conditions,
    while the closed-form dispersion w_k (Eq. 3) is derived assuming
    PERIODIC boundary conditions. So exact agreement is *not* expected at
    finite N -- only convergence as N grows, since boundary effects become
    relatively less important. This test checks both:
      (1) the relative error stays within a loose bound at moderate N, and
      (2) the error strictly decreases as N increases (finite-size scaling),
    which together validate that the real-space Eq. 9 Hamiltonian and the
    free-mode dispersion (Eq. 3) are mutually consistent, and that the
    mismatch is a genuine finite-size/boundary artifact rather than a bug.
    """
    Ns = [8, 12, 16]
    m = 0.3
    rel_errors = []
    for N in Ns:
        H = ts.build_hamiltonian(N, m, g=0.0)
        vals, _ = eigsh(H, k=1, which="SA")
        E_numeric = vals[0]

        k = ts.momentum_grid(N)
        E_analytic = -np.sum(ts.w_k(k, m))

        rel_err = abs(E_numeric - E_analytic) / abs(E_analytic)
        rel_errors.append(rel_err)
        assert rel_err < 0.06, (
            f"Free-theory vacuum energy mismatch too large for N={N}, m={m}: "
            f"numeric={E_numeric:.6f}, analytic={E_analytic:.6f}, rel_err={rel_err:.2%}"
        )

    assert all(rel_errors[i] > rel_errors[i + 1] for i in range(len(rel_errors) - 1)), (
        f"Expected the OBC-vs-PBC finite-size error to shrink monotonically with N, "
        f"got {rel_errors}"
    )
    print(
        "OK: free-theory (g=0) vacuum energy converges to the closed-form Dirac-sea "
        f"result as N grows (rel. errors {[f'{e:.2%}' for e in rel_errors]}); "
        "residual mismatch is the expected OBC-vs-PBC finite-size effect."
    )


def test_wavepacket_norm_reasonable():
    """
    C^dagger(phi^c) D^dagger(phi^d) |Omega> should be close to normalized
    (exactly 1 in the strict free-fermion limit with perfectly orthogonal
    modes; in the interacting theory and on a finite lattice we only expect
    it to be *close* to 1, as noted in the paper).
    """
    N, m, g = 16, 0.4, 0.5
    H = ts.build_hamiltonian(N, m, g)
    Omega = ts.ground_state(H)

    two_pi_over_N = 2 * np.pi / N
    Cdag = ts.wavepacket_operator(N, m, mu_k=4 * two_pi_over_N, mu_n=N / 4,
                                   sigma_k=two_pi_over_N, particle="c")
    Ddag = ts.wavepacket_operator(N, m, mu_k=-4 * two_pi_over_N, mu_n=3 * N / 4 - 1,
                                   sigma_k=two_pi_over_N, particle="d")
    psi0 = Ddag @ (Cdag @ Omega)
    norm = np.linalg.norm(psi0)
    assert 0.8 < norm <= 1.01, f"Unexpected wave-packet norm: {norm}"
    print(f"OK: wave-packet initial state norm = {norm:.4f} (expected close to 1).")


def test_entanglement_entropy_bounds():
    """Entropy of any single-qubit cut can't exceed 1 bit (log2(2))."""
    N = 8
    H = ts.build_hamiltonian(N, m=0.4, g=0.5)
    Omega = ts.ground_state(H)
    S1 = ts.bipartite_entropy(Omega, N, n_cut=1)
    assert 0 <= S1 <= 1.0 + 1e-8, f"S_1 out of bounds: {S1}"
    print(f"OK: single-qubit cut entropy S_1={S1:.4f} within [0, 1] bits.")


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} tests passed.")
