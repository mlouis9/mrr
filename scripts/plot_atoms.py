#!/usr/bin/env python3
"""Visualize the evolution of an MRR empirical source from CSV snapshots.

Reads the files written by the `sood` example (see its doc comment):

    atoms.csv  cycle,z,w,q0   one row per atom per cycle (cycle 0 = initial)
    k.csv      cycle,k        eigenvalue estimate after each cycle
    meta.csv   key,value      slab bounds and run parameters

and draws four panels:

(a) Atom positions z_i (horizontal) per cycle (vertical), coloured by q_i0.
(b) The source shape, two ways, on the same scale:
    - dots: q_i0 against z_i for a few cycles (light → dark with cycle).
      Since q_i0 = Q̂0(z_i), each atom is a pointwise estimate of Q0.
    - black: the binned density  Σ_{z_i ∈ bin} w_i q_i0 / Δz,  averaged over
      active cycles, with a ±1 standard-error band (cycle-to-cycle, ignoring
      autocorrelation, so the band is optimistic). It estimates the bin
      average of Q0 for any proposal, because w_i = 1/(N p(z_i)).
    The source is normalized so that ∫ Q0 dz = V, i.e. its mean is 1.
(c) Normalized effective sample size  ESS/N = (Σ c_i)² / (N Σ c_i²),
    c_i = w_i q_i0, per cycle. 1 means every atom carries equal source mass.
    With the uniform proposal w_i = V/N is constant, so all variation comes
    from q_i0.
(d) k per cycle, the running mean over active cycles, and the reference.

Usage:
    python3 scripts/plot_atoms.py out/sood [--inactive 10] [--bins 40] [-o fig.png]
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors


def load(run_dir: Path):
    atoms = np.genfromtxt(run_dir / "atoms.csv", delimiter=",", names=True)
    k = np.genfromtxt(run_dir / "k.csv", delimiter=",", names=True, ndmin=1)
    meta = dict(np.genfromtxt(run_dir / "meta.csv", delimiter=",", dtype=str, skip_header=1))
    return atoms, k, meta


def binned_density(z, c, edges):
    """Σ c_i over each bin divided by the bin width."""
    mass, _ = np.histogram(z, bins=edges, weights=c)
    return mass / np.diff(edges)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--inactive", type=int, default=10, help="cycles discarded before averaging")
    ap.add_argument("--bins", type=int, default=40)
    ap.add_argument("-o", "--output", type=Path, help="image file (default: <run_dir>/atoms.png)")
    args = ap.parse_args()

    atoms, k, meta = load(args.run_dir)
    lower, upper = float(meta["lower"]), float(meta["upper"])
    width = upper - lower
    k_ref = float(meta["k_ref"])
    cycles = np.unique(atoms["cycle"]).astype(int)
    last = cycles.max()
    if args.inactive >= last:
        raise SystemExit(f"--inactive {args.inactive} leaves no active cycles (last is {last})")

    edges = np.linspace(lower, upper, args.bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    # Sequential single hue for cycle; start at 30% so cycle 0 stays visible
    cmap = colors.LinearSegmentedColormap.from_list(
        "cycles", plt.get_cmap("Blues")(np.linspace(0.3, 1.0, 256)))
    cycle_norm = colors.Normalize(vmin=0, vmax=last)

    fig, ax = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    (a, b), (c_ax, d) = ax

    # (a) positions per cycle, coloured by q0
    q0 = atoms["q0"]
    sc = a.scatter(atoms["z"], atoms["cycle"], c=q0, s=2, cmap="viridis", linewidths=0, rasterized=True)
    fig.colorbar(sc, ax=a, label=r"$q_{i0}$")
    a.set(xlim=(lower, upper), xlabel="z (cm)", ylabel="cycle", title="(a) atom positions and $q_{i0}$")
    a.invert_yaxis()

    # (b) pointwise q0 for a few cycles + binned active mean
    shown = sorted({0, 1, 2, 5, args.inactive, last})
    for cyc in shown:
        sel = atoms["cycle"] == cyc
        b.scatter(atoms["z"][sel], q0[sel], s=4, color=cmap(cycle_norm(cyc)), linewidths=0,
                  label=f"cycle {cyc}", rasterized=True)
    active = np.array([
        binned_density(atoms["z"][sel], atoms["w"][sel] * q0[sel], edges)
        for sel in (atoms["cycle"] == cyc for cyc in cycles if cyc > args.inactive)
    ])
    mean = active.mean(axis=0)
    se = active.std(axis=0, ddof=1) / np.sqrt(len(active))
    b.fill_between(centres, mean - se, mean + se, step="mid", color="black", alpha=0.2, lw=0)
    b.step(centres, mean, where="mid", color="black", lw=1.5,
           label=fr"binned $\sum w_i q_{{i0}}/\Delta z$, cycles > {args.inactive}")
    b.axhline(1.0, color="gray", ls=":", lw=1)
    b.set(xlim=(lower, upper), ylim=(0, None), xlabel="z (cm)", ylabel=r"$Q_0$",
          title=r"(b) source shape: $q_{i0}$ and binned estimate")
    b.legend(loc="lower center", frameon=False, fontsize=8, ncol=2, markerscale=3)

    # (c) effective sample size of the source mass per cycle
    ess = []
    for cyc in cycles:
        sel = atoms["cycle"] == cyc
        ci = atoms["w"][sel] * q0[sel]
        ess.append(ci.sum() ** 2 / (len(ci) * (ci**2).sum()))
    c_ax.plot(cycles, ess, color="tab:blue", lw=2, marker="o", ms=3)
    c_ax.set(ylim=(0, 1.05), xlabel="cycle", ylabel="ESS / N",
             title=r"(c) effective sample size of $c_i = w_i q_{i0}$")
    c_ax.grid(alpha=0.3)

    # (d) k per cycle with running mean over active cycles
    kc, kv = k["cycle"].astype(int), k["k"]
    d.plot(kc, kv, color="tab:blue", lw=1, marker="o", ms=3, label="k per cycle")
    act = kc > args.inactive
    running = np.cumsum(kv[act]) / np.arange(1, act.sum() + 1)
    d.plot(kc[act], running, color="black", lw=2, label="running mean (active)")
    d.axhline(k_ref, color="gray", ls="--", lw=1, label=f"reference k = {k_ref:g}")
    d.axvline(args.inactive + 0.5, color="gray", ls=":", lw=1)
    d.set(xlabel="cycle", ylabel="k", title="(d) eigenvalue")
    d.grid(alpha=0.3)
    d.legend(frameon=False)

    fig.suptitle(f"{meta.get('problem', 'MRR run')}: N = {meta['n_atoms']}, seed {meta['seed']}; "
                 f"mean k (active) = {running[-1]:.4f}")
    out = args.output or args.run_dir / "atoms.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
