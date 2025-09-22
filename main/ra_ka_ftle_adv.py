# phase2_ftle_vs_margin.py
"""
Phase 2:  FTLE  ↔  Adversarial Margin
------------------------------------

* Uses Phase-1 checkpoints saved in rk_ckpts_v4/.
* If a checkpoint is missing, builds it by calling verify_or_train_checkpoint.
* For each test point:
    – Finds the smallest ℓ∞ PGD perturbation ε* that flips the label
      (10-step logarithmic bisection, ε ∈ [0, 0.30]).
    – Looks up the pre-computed FTLE value λ₁(x) on a cached grid.
* Reports Spearman ρ(λ₁, ε*) and plots density scatter + quartile AUC curves.
"""

import os, math, random, numpy as np, torch, torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import spearmanr

# ────────────────────────────────────────────────────────────────────
# Import Phase-1 objects
# ────────────────────────────────────────────────────────────────────
from ra_ka_best_method import (
    FC, make_circle,
    verify_or_train_checkpoint,   # used only if checkpoint missing
    dataset_to_loader,            # to build a train loader for verify
    DEVICE, BASE_LR, TRAIN_ACC_TARGET, MAX_EPOCHS, BATCH_SIZE_TRAIN
)

device = DEVICE

# ────────────────────────────────────────────────────────────────────
# Paths / caching
# ────────────────────────────────────────────────────────────────────
CHK_DIR  = "rk_ckpts_v4"   # Phase-1 saves here: model_N{N}_L{L}_seed{seed}.pt
FTLE_DIR = "ftle"          # cached FTLE grids
PLOT_DIR = "plots"
os.makedirs(CHK_DIR,  exist_ok=True)
os.makedirs(FTLE_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

def ckpt_path(N: int, L: int, seed: int) -> str:
    return os.path.join(CHK_DIR, f"model_N{N}_L{L}_seed{seed}.pt")

# ────────────────────────────────────────────────────────────────────
# 1) Load (or auto-build) checkpoint
# ────────────────────────────────────────────────────────────────────
def load_net(N: int, L: int, seed: int) -> FC:
    path = ckpt_path(N, L, seed)
    net = FC(N, L).to(device)
    if os.path.exists(path):
        state = torch.load(path, map_location=device)
        net.load_state_dict(state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state)
        net.eval()
        return net

    # If missing, auto-train via Phase-1 helper (fast if your Phase-1 already ran)
    print(f"[phase2] checkpoint missing → building: N={N} L={L} seed={seed}")
    (xt, yt), _ = make_circle()
    train_loader = dataset_to_loader((xt, yt), BATCH_SIZE_TRAIN, shuffle=True, device=device)
    net = verify_or_train_checkpoint(
        N, L, seed, train_loader,
        acc_target=TRAIN_ACC_TARGET,
        base_lr=BASE_LR,
        max_epochs=MAX_EPOCHS
    )
    return net

# ────────────────────────────────────────────────────────────────────
# 2) FTLE grid (cached)
# ────────────────────────────────────────────────────────────────────
def ftle_grid_path(N: int, L: int, seed: int, grid: int) -> str:
    return os.path.join(FTLE_DIR, f"ftle_N{N}_L{L}_seed{seed}_g{grid}.npy")

@torch.no_grad()
def ftle_field(net: FC, depth: int, grid: int = 161, bbox = (-1.2, 1.2)) -> np.ndarray:
    xs = torch.linspace(*bbox, grid, device=device)
    ys = torch.linspace(*bbox, grid, device=device)
    field = np.empty((grid, grid), np.float32)
    for i, xv in enumerate(xs):
        for j, yv in enumerate(ys):
            # per-point Jacobian of last-hidden activations wrt input
            x = torch.tensor([xv, yv], requires_grad=True, device=device)
            J = torch.autograd.functional.jacobian(
                    lambda z: net(z.unsqueeze(0), hid=True).squeeze(0), x)
            sigmax = torch.linalg.svdvals(J).max()
            # normalize by depth (finite-time Lyapunov per-layer scale)
            field[j, i] = (1.0 / depth) * math.log(float(sigmax) + 1e-12)
    return field

def load_ftle_grid(N: int, L: int, seed: int, grid: int = 161) -> np.ndarray:
    path = ftle_grid_path(N, L, seed, grid)
    if os.path.exists(path):
        return np.load(path)
    net = load_net(N, L, seed)
    fld = ftle_field(net, L, grid)
    np.save(path, fld)
    return fld

def ftle_lookup(fld: np.ndarray, x: np.ndarray, bbox = (-1.2, 1.2)) -> float:
    gx, gy = fld.shape[1] - 1, fld.shape[0] - 1
    xmin, xmax = bbox
    i = int((x[0] - xmin) / (xmax - xmin) * gx)
    j = int((x[1] - xmin) / (xmax - xmin) * gy)
    i = np.clip(i, 0, gx); j = np.clip(j, 0, gy)
    return float(fld[j, i])

# ────────────────────────────────────────────────────────────────────
# 3) PGD attack + log-bisection margin
# ────────────────────────────────────────────────────────────────────
def pgd(net: FC, x: torch.Tensor, y: torch.Tensor, eps: float, step: float, k: int = 20):
    delta = torch.zeros_like(x, requires_grad=True)
    for _ in range(k):
        out = net(x + delta)
        loss = - (y * out).mean()
        loss.backward()
        delta.data = (delta + step * delta.grad.sign()).clamp(-eps, eps)
        delta.grad.zero_()
    return (x + delta).detach()

def margin(net: FC, x: torch.Tensor, y: torch.Tensor, eps_hi: float = 0.30) -> float:
    lo, hi = 0.0, eps_hi
    for _ in range(10):  # ~1e-3 precision
        mid = 0.5 * (lo + hi)
        adv = pgd(net, x, y, mid, step=mid/10)
        if torch.sign(net(adv)) != y:
            hi = mid
        else:
            lo = mid
    return hi

# ────────────────────────────────────────────────────────────────────
# 4) Evaluate one (N, L) config over seeds
# ────────────────────────────────────────────────────────────────────
def evaluate_model(N: int, L: int, seeds, eps_hi: float = 0.30, grid: int = 161):
    (xt, yt), (xe, ye) = make_circle()
    test_loader = DataLoader(TensorDataset(xe.to(device), ye.to(device)),
                             batch_size=1024, shuffle=False)
    margins, ftle_vals = [], []
    for sd in seeds:
        net = load_net(N, L, sd)
        fld = load_ftle_grid(N, L, sd, grid)
        for xb, yb in tqdm(test_loader, desc=f"seed {sd}  N={N} L={L}", leave=False):
            for xi, yi in zip(xb, yb):
                eps_star = margin(net, xi.unsqueeze(0), yi, eps_hi)
                margins.append(eps_star)
                ftle_vals.append(ftle_lookup(fld, xi.cpu().numpy()))
    return np.array(margins, dtype=np.float32), np.array(ftle_vals, dtype=np.float32)

# ────────────────────────────────────────────────────────────────────
# 5) Spearman ρ + plots
# ────────────────────────────────────────────────────────────────────
def summarize_and_plot(N: int, L: int, seeds):
    m, f = evaluate_model(N, L, seeds)
    rho, p = spearmanr(f, m)
    print(f"N={N:<4} L={L:<3}  Spearman ρ = {rho:.3f}  (p={p:.1e})")

    # density scatter
    plt.figure(figsize=(4,4))
    plt.hexbin(f, m, gridsize=60, cmap="magma", mincnt=1)
    plt.xlabel("λ₁ (FTLE)"); plt.ylabel("margin ε*")
    plt.title(f"N={N}, L={L}, ρ={rho:.3f}")
    plt.colorbar(label="count")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"ftle_vs_margin_N{N}_L{L}.png"), dpi=300)

    # quartile AUC curves
    q = np.quantile(f, [0.25, 0.50, 0.75])
    idx = np.digitize(f, q)   # 0..3
    eps_grid = np.arange(0.0, 0.31, 0.02)
    plt.figure(figsize=(4,3))
    for qbin in range(4):
        succ = [(m[idx==qbin] <= e).mean() for e in eps_grid]
        plt.plot(eps_grid, succ, label=f"Q{qbin+1}")
    plt.legend();  plt.xlabel("ε");  plt.ylabel("attack success")
    plt.title(f"AUC by FTLE quartile  (N={N}, L={L})")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"auc_quartiles_N{N}_L{L}.png"), dpi=300)

    return rho, p

# ────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.set_grad_enabled(True)

    seeds = list(range(9))

    # Example configs: rich vs lazy from Phase-1 grid
    rich_cfg = (10, 12)    # narrow, deep
    lazy_cfg = (250, 2)    # wide, shallow

    rho_rich, p_rich = summarize_and_plot(*rich_cfg, seeds)
    rho_lazy, p_lazy = summarize_and_plot(*lazy_cfg, seeds)

    print("\n---------- summary ----------")
    print(f"rich  (N={rich_cfg[0]} × L={rich_cfg[1]})   ρ = {rho_rich:.3f}  (p={p_rich:.1e})")
    print(f"lazy  (N={lazy_cfg[0]} × L={lazy_cfg[1]})   ρ = {rho_lazy:.3f}  (p={p_lazy:.1e})")


##### Result from Sep 25th #####
# rich  (N=10 × L=12)   ρ = -0.973  (p=0.0e+00)
# lazy  (N=250 × L=2)   ρ = 0.199  (p=0.0e+00)

