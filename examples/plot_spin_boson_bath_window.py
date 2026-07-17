from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


root = Path(__file__).resolve().parents[1]
output_dir = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else root / "results" / "bath_quench"
stem = sys.argv[2] if len(sys.argv) > 2 else "bath_quench_sudden_turn_on"
ramp_stop = float(sys.argv[3]) if len(sys.argv) > 3 else None
data = np.genfromtxt(output_dir / f"{stem}_observables.csv", delimiter=",", names=True)

fig, axes = plt.subplots(4, 1, figsize=(8.0, 10.5), sharex=True, constrained_layout=True)

axes[0].plot(data["t"], data["Sx"], label=r"$\langle S_x\rangle$")
axes[0].plot(data["t"], data["Sy"], label=r"$\langle S_y\rangle$")
axes[0].plot(data["t"], data["Sz"], label=r"$\langle S_z\rangle$")
axes[0].set_ylabel("Spin component")
axes[0].legend(ncol=3, frameon=False)

axes[1].plot(data["t"], data["spin_norm"], color="black")
axes[1].axhline(0.5, color="0.55", linestyle="--", linewidth=1, label=r"$S=1/2$")
axes[1].set_ylabel(r"$|\langle \mathbf{S}\rangle|$")
axes[1].legend(frameon=False)

axes[2].plot(data["t"], data["boson_constraint"], color="#0072B2")
axes[2].axhline(1.0, color="0.55", linestyle="--", linewidth=1, label=r"$2S=1$")
axes[2].set_ylabel(r"$\langle n_a+n_b\rangle$")
axes[2].legend(frameon=False)

dt = np.diff(data["t"])
axes[3].plot(data["t"][1:], dt, color="#009E73", linewidth=1)
axes[3].set_ylabel(r"$\Delta t$")
axes[3].set_xlabel("Time")

for axis in axes:
    axis.axvspan(7.0, 14.0, color="#E69F00", alpha=0.12)
    axis.axvline(7.0, color="#E69F00", linewidth=0.8)
    if ramp_stop is not None:
        axis.axvline(ramp_stop, color="#E69F00", linestyle="--", linewidth=0.8)
    axis.axvline(14.0, color="#E69F00", linewidth=0.8)
    axis.grid(alpha=0.2)

figure_path = output_dir / f"{stem}.png"
fig.savefig(figure_path, dpi=180)
print(figure_path)
