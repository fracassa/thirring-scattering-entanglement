"""
thirring_sim.py

Essential (Level 1) classical replication of the entanglement-growth result
(Fig. 2) from:

    Chai, Gibbs, Pascuzzi, Holmes, Kuehn, Tacchino, Tavernelli,
    "Resource-Efficient Simulations of Particle Scattering on a
    Digital Quantum Computer", arXiv:2507.17832 (2025).

This is a small exact-diagonalization (full statevector) simulation of the
lattice Thirring model, on N <= ~18 qubits. It does NOT use tensor networks
or a quantum circuit -- it reproduces the *physics* (fermion/antifermion
wave-packet scattering and the resulting entanglement growth) which the
paper's tensor-network circuit-compression method is built to exploit.

Equation numbers in comments refer to the paper above.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, expm_multiply

# ----------------------------------------------------------------------
# Single-qubit operators
# ----------------------------------------------------------------------

I2 = sp.identity(2, format="csr", dtype=complex)
X = sp.csr_matrix(np.array([[0, 1], [1, 0]], dtype=complex))
Y = sp.csr_matrix(np.array([[0, -1j], [1j, 0]], dtype=complex))
Z = sp.csr_matrix(np.array([[1, 0], [0, -1]], dtype=complex))
SIGMA_PLUS = (X + 1j * Y) / 2   # raises: |0><1|  (with |0> = up = +1 eigval of Z... see note)
SIGMA_MINUS = (X - 1j * Y) / 2  # lowers: |1><0|


def multi_site_op(N: int, site_ops: dict[int, sp.spmatrix]) -> sp.spmatrix:
    """Build an N-qubit sparse operator: `site_ops[i]` acting on qubit i,
    identity on every other qubit."""
    mats = [site_ops.get(n, I2) for n in range(N)]
    out = mats[0]
    for m in mats[1:]:
        out = sp.kron(out, m, format="csr")
    return out


def xi_dagger(n: int, N: int) -> sp.spmatrix:
    """Jordan-Wigner fermion creation operator (Eq. 8):
    xi_n^dagger = (prod_{l<n} sigma^z_l) sigma^-_n
    """
    ops = {l: Z for l in range(n)}
    ops[n] = SIGMA_MINUS
    return multi_site_op(N, ops)


def xi(n: int, N: int) -> sp.spmatrix:
    """Jordan-Wigner fermion annihilation operator (Eq. 8):
    xi_n = (prod_{l<n} sigma^z_l) sigma^+_n
    """
    ops = {l: Z for l in range(n)}
    ops[n] = SIGMA_PLUS
    return multi_site_op(N, ops)


# ----------------------------------------------------------------------
# Hamiltonian (Eq. 9), open boundary conditions
# ----------------------------------------------------------------------

def build_hamiltonian(N: int, m: float, g: float) -> sp.spmatrix:
    """
    H = (i/2) sum_{n=0}^{N-2} (sigma^-_{n+1} sigma^+_n - sigma^-_n sigma^+_{n+1})
      + (m/2) sum_{n=0}^{N-1} (-1)^n (I - sigma^z_n)
      + (g/4) sum_{n=0}^{N-2} (I - sigma^z_n)(I - sigma^z_{n+1})
    """
    dim = 2 ** N
    H = sp.csr_matrix((dim, dim), dtype=complex)

    # Hopping term
    for n in range(N - 1):
        term1 = multi_site_op(N, {n + 1: SIGMA_MINUS, n: SIGMA_PLUS})
        term2 = multi_site_op(N, {n: SIGMA_MINUS, n + 1: SIGMA_PLUS})
        H = H + (1j / 2) * (term1 - term2)

    # Staggered mass term
    for n in range(N):
        H = H + (m / 2) * ((-1) ** n) * (multi_site_op(N, {}) - multi_site_op(N, {n: Z}))

    # Four-fermion interaction term
    for n in range(N - 1):
        IminusZn = multi_site_op(N, {}) - multi_site_op(N, {n: Z})
        IminusZn1 = multi_site_op(N, {}) - multi_site_op(N, {n + 1: Z})
        H = H + (g / 4) * (IminusZn @ IminusZn1)

    # H should be Hermitian; symmetrize away tiny numerical asymmetry
    H = (H + H.getH()) / 2
    return H.tocsr()


def ground_state(H: sp.spmatrix) -> np.ndarray:
    """Lowest-energy eigenstate of H via Lanczos (ARPACK)."""
    vals, vecs = eigsh(H, k=1, which="SA")
    psi = vecs[:, 0]
    return psi / np.linalg.norm(psi)


# ----------------------------------------------------------------------
# Free-theory momentum modes (Eq. 2-4) -- used only to build the
# wave-packet creation operators in real space (Eq. 6-7 combined with Eq. 2).
# ----------------------------------------------------------------------

def momentum_grid(N: int) -> np.ndarray:
    if N % 4 != 0:
        raise ValueError("N must be a multiple of 4 (paper's momentum grid convention).")
    idx = np.arange(-(N // 4), N // 4)  # N/2 values
    return 2 * np.pi / N * idx


def w_k(k: np.ndarray, m: float) -> np.ndarray:
    return np.sqrt(m ** 2 + np.sin(k) ** 2)


def v_k(k: np.ndarray, m: float) -> np.ndarray:
    return np.sin(k) / (m + w_k(k, m))


def proj(n: int, l: int) -> float:
    """Pi_{n,l} = (1 + (-1)^{n+l}) / 2"""
    return (1 + (-1) ** (n + l)) / 2


def gaussian_wavepacket_k(N: int, mu_k: float, mu_n: float, sigma_k: float) -> np.ndarray:
    """phi_k (Eq. 7), normalized so sum_k |phi_k|^2 = 1."""
    k = momentum_grid(N)
    phi = np.exp(-1j * k * mu_n) * np.exp(-((k - mu_k) ** 2) / (4 * sigma_k ** 2))
    phi /= np.linalg.norm(phi)
    return phi


def real_space_coeffs(N: int, m: float, phi_k: np.ndarray, particle: str) -> np.ndarray:
    """
    Combine Eq. 2 and Eq. 6 to get the real-space coefficients Phi_n such
    that   sum_k phi_k * c_k^dagger  =  sum_n Phi_n * xi_n^dagger      (particle="c")
           sum_k phi_k * d_k^dagger  =  sum_n Phi_n * xi_n             (particle="d")
    """
    k = momentum_grid(N)
    wk = w_k(k, m)
    vk = v_k(k, m)
    amp = np.sqrt((m + wk) / wk)  # shape (Nk,)

    Phi = np.zeros(N, dtype=complex)
    for n in range(N):
        if particle == "c":
            proj_term = proj(n, 0) + vk * proj(n, 1)
        elif particle == "d":
            proj_term = proj(n, 1) + vk * proj(n, 0)
        else:
            raise ValueError("particle must be 'c' or 'd'")
        Phi[n] = np.sum(phi_k * amp * np.exp(1j * k * n) * proj_term) / np.sqrt(N)
    return Phi


def wavepacket_operator(N: int, m: float, mu_k: float, mu_n: float, sigma_k: float,
                         particle: str) -> sp.spmatrix:
    """Build C^dagger(phi^c) or D^dagger(phi^d) as an N-qubit sparse operator."""
    phi_k = gaussian_wavepacket_k(N, mu_k, mu_n, sigma_k)
    Phi_n = real_space_coeffs(N, m, phi_k, particle)

    dim = 2 ** N
    op = sp.csr_matrix((dim, dim), dtype=complex)
    for n in range(N):
        if abs(Phi_n[n]) < 1e-14:
            continue
        base = xi_dagger(n, N) if particle == "c" else xi(n, N)
        op = op + Phi_n[n] * base
    return op


# ----------------------------------------------------------------------
# Entanglement entropy
# ----------------------------------------------------------------------

def bipartite_entropy(psi: np.ndarray, N: int, n_cut: int) -> float:
    """Von Neumann entropy (in bits) of the reduced state of the first
    n_cut qubits, given the full statevector psi of N qubits."""
    dimA = 2 ** n_cut
    dimB = 2 ** (N - n_cut)
    mat = psi.reshape(dimA, dimB)
    svals = np.linalg.svd(mat, compute_uv=False)
    p = svals ** 2
    p = p[p > 1e-14]
    return float(-np.sum(p * np.log2(p)))


def all_cuts_entropy(psi: np.ndarray, N: int) -> np.ndarray:
    return np.array([bipartite_entropy(psi, N, n) for n in range(1, N)])
