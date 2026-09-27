"""
run.py -- reproduce (in essential form) Fig. 2 of arXiv:2507.17832:
entanglement entropy growth during fermion-antifermion scattering in the
lattice Thirring model.

Usage:
    python run.py --N 16 --m 0.4 --g 0.5 --T 24 --steps 40
"""

import argparse
import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import expm_multiply

import thirring_sim as ts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--N", type=int, default=16, help="number of qubits/sites (multiple of 4)")
    parser.add_argument("--m", type=float, default=0.4, help="fermion mass")
    parser.add_argument("--g", type=float, default=0.5, help="four-fermion coupling")
    parser.add_argument("--T", type=float, default=24.0, help="total evolution time")
    parser.add_argument("--steps", type=int, default=40, help="number of time steps")
    parser.add_argument("--outdir", type=str, default="results", help="output directory")
    args = parser.parse_args()

    N, m, g, T, steps = args.N, args.m, args.g, args.T, args.steps

    print(f"[1/5] Building Hamiltonian (N={N}, m={m}, g={g}) ...")
    t0 = time.time()
    H = ts.build_hamiltonian(N, m, g)
    print(f"      done in {time.time()-t0:.2f}s, dim = {H.shape[0]}")

    print("[2/5] Finding ground state (Lanczos) ...")
    t0 = time.time()
    Omega = ts.ground_state(H)
    print(f"      done in {time.time()-t0:.2f}s")

    print("[3/5] Building wave-packet creation operators C^dagger, D^dagger ...")
    t0 = time.time()
    two_pi_over_N = 2 * np.pi / N
    Cdag = ts.wavepacket_operator(
        N, m,
        mu_k=4 * two_pi_over_N, mu_n=N / 4, sigma_k=two_pi_over_N,
        particle="c",
    )
    Ddag = ts.wavepacket_operator(
        N, m,
        mu_k=-4 * two_pi_over_N, mu_n=3 * N / 4 - 1, sigma_k=two_pi_over_N,
        particle="d",
    )
    psi0 = Ddag @ (Cdag @ Omega)
    norm0 = np.linalg.norm(psi0)
    if norm0 < 1e-10:
        raise RuntimeError("Wave-packet initial state has ~zero norm; check parameters.")
    psi0 = psi0 / norm0
    print(f"      done in {time.time()-t0:.2f}s (initial-state norm before normalization: {norm0:.4f})")

    print(f"[4/5] Time evolution over T={T} in {steps} steps ...")
    t0 = time.time()
    times = np.linspace(0, T, steps)
    # expm_multiply with start/stop/num returns the full trajectory efficiently
    trajectory = expm_multiply(-1j * H, psi0, start=0, stop=T, num=steps)
    print(f"      done in {time.time()-t0:.2f}s")

    print("[5/5] Computing entanglement entropy at each time / cut ...")
    t0 = time.time()
    S_vacuum = ts.all_cuts_entropy(Omega, N)  # subtract vacuum contribution
    dS_grid = np.zeros((steps, N - 1))
    for i in range(steps):
        psi_t = trajectory[i]
        S_t = ts.all_cuts_entropy(psi_t, N)
        dS_grid[i, :] = S_t - S_vacuum
    dS_total = dS_grid.sum(axis=1)
    print(f"      done in {time.time()-t0:.2f}s")

    # ------------------------------------------------------------------
    # Plotting (reproduces the structure of Fig. 2 in the paper)
    # ------------------------------------------------------------------
    import os
    os.makedirs(args.outdir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Panel (a): heatmap of Delta S_n(t)
    ax = axes[0]
    site_idx = np.arange(1, N)
    im = ax.pcolormesh(site_idx, times, dS_grid, shading="auto", cmap="viridis")
    fig.colorbar(im, ax=ax, label=r"$\Delta S_n(t)$")
    ax.set_xlabel("site index $n$")
    ax.set_ylabel("time $t$")
    ax.set_title(f"(a) Entanglement entropy per cut\n(m={m}, g={g}, N={N})")

    # Panel (b): total excess entropy vs time
    ax = axes[1]
    ax.plot(times, dS_total, marker="o", ms=3)
    ax.set_xlabel("time $t$")
    ax.set_ylabel(r"$\Delta S(t) = \sum_n \Delta S_n(t)$")
    ax.set_title("(b) Total excess entanglement entropy")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    outpath = os.path.join(args.outdir, f"entanglement_growth_N{N}_m{m}_g{g}.png")
    fig.savefig(outpath, dpi=150)
    print(f"\nSaved figure to {outpath}")

    np.savez(
        os.path.join(args.outdir, f"data_N{N}_m{m}_g{g}.npz"),
        times=times, dS_grid=dS_grid, dS_total=dS_total, N=N, m=m, g=g,
    )
    print(f"Saved raw data to {args.outdir}/data_N{N}_m{m}_g{g}.npz")


if __name__ == "__main__":
    main()
