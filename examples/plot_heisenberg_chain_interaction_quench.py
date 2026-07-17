from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


length_error = "Usage: plot_heisenberg_chain_interaction_quench.py sudden|linear|smooth"
if len(sys.argv) < 2:
    raise SystemExit(length_error)

kind = sys.argv[1].lower()
if kind not in {"sudden", "linear", "smooth"}:
    raise SystemExit(length_error)

root = Path(__file__).resolve().parents[1]
stem = f"interaction_quench_{kind}_turn_on"
output_dir = root / "results" / "interaction_quench"
data = np.genfromtxt(output_dir / f"{stem}_observables.csv", delimiter=",", names=True)

fig, axes = plt.subplots(4, 1, figsize=(8.0, 10.5), sharex=True, constrained_layout=True)

for axis_name, label in zip(("x", "y", "z"), (r"$L_x$", r"$L_y$", r"$L_z$")):
    axes[0].plot(data["t"], data[f"staggered_{axis_name}"], label=label)
axes[0].set_ylabel("Staggered spin")
axes[0].legend(ncol=3, frameon=False)

for n in range(1, 5):
    axes[1].plot(data["t"], data[f"spin_norm{n}"], label=fr"site {n}")
axes[1].axhline(0.5, color="0.55", linestyle="--", linewidth=1, label=r"$S=1/2$")
axes[1].set_ylabel(r"$|\langle \mathbf{S}_n\rangle|$")
axes[1].legend(ncol=5, frameon=False)

for n in range(1, 5):
    axes[2].plot(data["t"], data[f"boson_constraint{n}"], label=fr"site {n}")
axes[2].axhline(1.0, color="0.55", linestyle="--", linewidth=1, label=r"$2S=1$")
axes[2].set_ylabel(r"$\langle n_a+n_b\rangle$")
axes[2].legend(ncol=5, frameon=False)

dt = np.diff(data["t"])
axes[3].plot(data["t"][1:], dt, color="#009E73", linewidth=1)
axes[3].set_ylabel(r"$\Delta t$")
axes[3].set_xlabel("Time")

for axis in axes:
    axis.axvspan(7.0, 14.0, color="#E69F00", alpha=0.12)
    axis.axvline(7.0, color="#E69F00", linewidth=0.8)
    if kind != "sudden":
        axis.axvline(8.0, color="#E69F00", linestyle="--", linewidth=0.8)
    axis.axvline(14.0, color="#E69F00", linewidth=0.8)
    axis.grid(alpha=0.2)

figure_path = output_dir / f"{stem}.png"
fig.savefig(figure_path, dpi=180)
print(figure_path)
