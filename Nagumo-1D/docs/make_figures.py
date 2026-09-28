# Figures for ramsay_fhn_parametric_PINNs_implementation_details.tex, from the saved run data.
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
out = str(Path(__file__).resolve().parent / "figures") + "/"
d = np.load(out + "run_data.npz")
plt.rcParams.update({"font.size": 11, "font.family": "serif", "mathtext.fontset": "cm"})

fig, ax = plt.subplots(figsize=(6.4, 3.2))
for i, a in enumerate(d["a_train"]):
    ax.axvline(a, color="0.6", lw=0.5, label="training $a$" if i == 0 else None)
ax.semilogy(d["a_test"], d["rel_l2"], "k-", lw=1.6, label="$(V, R)$")
ax.semilogy(d["a_test"], d["rel_l2_V"], "--", lw=1.1, label="$V$")
ax.semilogy(d["a_test"], d["rel_l2_R"], "--", lw=1.1, label="$R$")
ax.set_xlabel("$a$"); ax.set_ylabel(r"relative $L^2$ error on $[0, 20]$")
ticks = [3e-4, 5e-4, 1e-3, 2e-3, 3e-3]
ax.set_yticks(ticks); ax.set_yticklabels([f"${t*1e3:g}\\times10^{{-3}}$" for t in ticks]); ax.minorticks_off()
ax.legend(loc="upper center", ncol=4, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 1.13))
ax.grid(alpha=0.3, which="both")
fig.tight_layout(); fig.savefig(out + "error_vs_a.pdf", bbox_inches="tight")

fig, ax = plt.subplots(figsize=(6.4, 3.2))
cmap = plt.get_cmap("viridis"); K = len(d["edges"]) - 1
for k in range(K):
    ax.semilogy(d[f"hist_{k}"], color=cmap(k / max(K - 1, 1)), lw=0.8)
ax.axvline(int(d["adam_epochs"]), color="k", ls=":", lw=1, label=r"Adam $\to$ L-BFGS")
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, K - 1)); fig.colorbar(sm, ax=ax, label="window index $k$")
ax.set_xlabel("iteration (Adam steps, then L-BFGS function evaluations)", fontsize=9.5)
ax.set_ylabel("residual loss $\\mathcal{L}_k$"); ax.legend(fontsize=9); ax.grid(alpha=0.3, which="both")
fig.tight_layout(); fig.savefig(out + "loss_history.pdf", bbox_inches="tight")
print("figs ok")
