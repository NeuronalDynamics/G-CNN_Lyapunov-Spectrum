# phase2_ftle_vs_margin.py
"""
Phase 2:  FTLE  ↔  Adversarial Margin  (+ Jacobian variance)
-----------------------------------------------------------

* Uses Phase-1 checkpoints saved via ra_ka_best_method_accstop.py.
* For each test point x:
    – Finds the smallest ℓ∞ PGD perturbation ε* that flips the label
      (log-bisection over ε ∈ [0, 0.30]).
    – Looks up the pre-computed FTLE value λ(x) on a cached grid.
* For each configuration (N, L, gain, base_lr):
    – Computes G_λ   = Var_x[λ(x)]
    – Computes G_J   = Var_x[||J(x)||_2], where ||J||_2 = exp(L * λ(x))
    – Computes Spearman ρ(λ, margin) and ρ(||J||, margin).
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import os, math, random, numpy as np, torch, torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import spearmanr

# ────────────────────────────────────────────────────────────────────
# Import Phase-1 objects
# ────────────────────────────────────────────────────────────────────
from ra_ka_best_method_accstop import (
    FC, make_circle,
    verify_or_train_checkpoint,
    dataset_to_loader,
    DEVICE, TRAIN_ACC_TARGET, MAX_EPOCHS, BATCH_SIZE_TRAIN,
    fmt_float,
)

device = DEVICE

# Width/depth grid (you can adjust)
WIDTHS  = [10, 50, 100, 150, 200]
DEPTHS  = [2, 6, 8, 10, 12, 14, 16]

# Gain & LR grids (centered around Phase-1 base settings)
GAINS    = [0.4, 0.6, 0.85, 1.0, 1.2, 1.6, 2.2]
BASE_LRS = [0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30]

# Paths / caching
FTLE_DIR = "ftle"
PLOT_DIR = "plots"
os.makedirs(FTLE_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


# ────────────────────────────────────────────────────────────────────
# 1) Load (or auto-build) checkpoint for a given (N,L,gain,base_lr,seed)
# ────────────────────────────────────────────────────────────────────
def load_net(N: int, L: int, gain: float, base_lr: float, seed: int) -> FC:
    """
    Delegate to Phase-1's verify_or_train_checkpoint, which already
    handles checkpointing keyed by (N, L, gain, base_lr, seed).
    """
    (xt, yt), _ = make_circle()
    train_loader = dataset_to_loader((xt, yt), BATCH_SIZE_TRAIN, shuffle=True, device=device)
    net = verify_or_train_checkpoint(
        N, L, gain, base_lr, seed,
        train_loader=train_loader,
        acc_target=TRAIN_ACC_TARGET,
        max_epochs=MAX_EPOCHS
    )
    return net


# ────────────────────────────────────────────────────────────────────
# 2) FTLE grid (cached)
# ────────────────────────────────────────────────────────────────────
def ftle_grid_path(N: int, L: int, gain: float, base_lr: float, seed: int, grid: int) -> str:
    gstr  = fmt_float(gain)
    lrstr = fmt_float(base_lr)
    return os.path.join(FTLE_DIR, f"ftle_N{N}_L{L}_g{gstr}_lr{lrstr}_seed{seed}_g{grid}.npy")


@torch.no_grad()
def ftle_field(net: FC, depth: int, grid: int = 161, bbox=(-1.2, 1.2)) -> np.ndarray:
    """
    Compute FTLE λ(x) on a regular grid over the input square,
    using the Jacobian of the last-hidden activations wrt input.
    """
    xs = torch.linspace(*bbox, grid, device=device)
    ys = torch.linspace(*bbox, grid, device=device)
    field = np.empty((grid, grid), np.float32)
    for i, xv in enumerate(xs):
        for j, yv in enumerate(ys):
            x = torch.tensor([xv, yv], requires_grad=True, device=device)
            J = torch.autograd.functional.jacobian(
                lambda z: net(z.unsqueeze(0), hid=True).squeeze(0), x
            )
            sigmax = torch.linalg.svdvals(J).max()
            # finite-time Lyapunov per layer
            field[j, i] = (1.0 / depth) * math.log(float(sigmax) + 1e-12)
    return field


def load_ftle_grid(N: int, L: int, gain: float, base_lr: float, seed: int, grid: int = 161) -> np.ndarray:
    path = ftle_grid_path(N, L, gain, base_lr, seed, grid)
    if os.path.exists(path):
        return np.load(path)
    net = load_net(N, L, gain, base_lr, seed)
    fld = ftle_field(net, L, grid)
    np.save(path, fld)
    return fld


def ftle_lookup(fld: np.ndarray, x: np.ndarray, bbox=(-1.2, 1.2)) -> float:
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
# 4) Evaluate one (N, L, gain, base_lr) config over seeds
# ────────────────────────────────────────────────────────────────────
def evaluate_model(N: int, L: int,
                   gain: float, base_lr: float,
                   seeds, eps_hi: float = 0.30, grid: int = 161):
    """
    For a given (N, L, gain, base_lr) and seed set,
    return arrays of margins ε*(x) and FTLE λ(x) on the test set.
    """
    (xt, yt), (xe, ye) = make_circle()
    test_loader = DataLoader(TensorDataset(xe.to(device), ye.to(device)),
                             batch_size=512, shuffle=False)
    margins, ftle_vals = [], []
    for sd in seeds:
        net = load_net(N, L, gain, base_lr, sd)
        fld = load_ftle_grid(N, L, gain, base_lr, sd, grid)
        for xb, yb in tqdm(test_loader,
                           desc=f"N={N} L={L} g={gain} lr={base_lr} seed={sd}",
                           leave=False):
            for xi, yi in zip(xb, yb):
                eps_star = margin(net, xi.unsqueeze(0), yi, eps_hi)
                margins.append(eps_star)
                ftle_vals.append(ftle_lookup(fld, xi.cpu().numpy()))
    margins   = np.array(margins,   dtype=np.float32)
    ftle_vals = np.array(ftle_vals, dtype=np.float32)
    return margins, ftle_vals


# ────────────────────────────────────────────────────────────────────
# 5) G_λ, G_J and correlations for a single config
# ────────────────────────────────────────────────────────────────────
def stats_for_config(N: int, L: int,
                     gain: float, base_lr: float,
                     seeds, eps_hi: float = 0.30, grid: int = 161):
    """
    For a given (N, L, gain, base_lr) and seeds:
      - compute margins ε*(x) and FTLE λ(x),
      - compute:
          G_lambda = Var_x[λ(x)]
          G_J      = Var_x[||J(x)||_2], where ||J||_2 = exp(L * λ(x))
      - compute Spearman ρ between λ and margins,
      - compute Spearman ρ between ||J||_2 and margins.
    """
    margins, ftle_vals = evaluate_model(N, L, gain, base_lr, seeds,
                                        eps_hi=eps_hi, grid=grid)

    # G_lambda: variance of FTLE over inputs
    G_lambda = float(np.var(ftle_vals))

    # Jacobian spectral norm from FTLE: σ_max = exp(L * λ)
    jac_norms = np.exp(L * ftle_vals)  # ||J(x)||_2 at each test point

    # G_J: variance of Jacobian spectral norm
    G_J = float(np.var(jac_norms))

    # Correlations
    rho_lambda, p_lambda = spearmanr(ftle_vals, margins)
    rho_J,      p_J      = spearmanr(jac_norms, margins)

    # Correlation between λ and ||J||_2 (should be very close to 1.0)
    corr_lam_J = float(np.corrcoef(ftle_vals, jac_norms)[0, 1])

    print(
        f"N={N:<4} L={L:<3} g={gain:.3g} lr={base_lr:.3g}  "
        f"G_lambda={G_lambda:.3e}  G_J={G_J:.3e}  "
        f"ρ(λ,m)={rho_lambda:.3f}  ρ(||J||,m)={rho_J:.3f}  "
        f"corr(λ,||J||)={corr_lam_J:.3f}"
    )

    return {
        "G_lambda": G_lambda,
        "G_J": G_J,
        "rho_lambda_margin": rho_lambda,
        "rho_J_margin": rho_J,
        "corr_lambda_J": corr_lam_J,
        "margins": margins,
        "ftle_vals": ftle_vals,
        "jac_norms": jac_norms,
    }


# ────────────────────────────────────────────────────────────────────
# 6) Optional: full 4D grid over (gain, lr, depth, width)
# ────────────────────────────────────────────────────────────────────
def run_full_grid(widths, depths, gains, base_lrs, seeds,
                  eps_hi: float = 0.30, grid: int = 161):
    """
    Heavy: runs stats_for_config on the full (gain, lr, depth, width) grid.
    Returns 4D arrays:
        G_lambda[gi, li, di, wi]
        G_J     [gi, li, di, wi]
        rho_lam [gi, li, di, wi]
        rho_J   [gi, li, di, wi]
    """
    shape = (len(gains), len(base_lrs), len(depths), len(widths))
    G_lambda_map = np.zeros(shape, dtype=np.float32)
    G_J_map      = np.zeros_like(G_lambda_map)
    rho_lam_map  = np.zeros_like(G_lambda_map)
    rho_J_map    = np.zeros_like(G_lambda_map)

    for gi, gain in enumerate(gains):
        for li, lr in enumerate(base_lrs):
            for di, L in enumerate(depths):
                for wi, N in enumerate(widths):
                    print(f"\n[config] N={N} L={L} gain={gain} lr={lr}")
                    stats = stats_for_config(N, L, gain, lr, seeds,
                                             eps_hi=eps_hi, grid=grid)
                    G_lambda_map[gi, li, di, wi] = stats["G_lambda"]
                    G_J_map[gi, li, di, wi]      = stats["G_J"]
                    rho_lam_map[gi, li, di, wi]  = stats["rho_lambda_margin"]
                    rho_J_map[gi, li, di, wi]    = stats["rho_J_margin"]

    return G_lambda_map, G_J_map, rho_lam_map, rho_J_map


# ────────────────────────────────────────────────────────────────────
# 7) Simple rich vs lazy demo + (optional) grid
# ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.set_grad_enabled(True)

    # Seeds for averaging
    seeds = list(range(3))   # reduce for speed in grid runs

    # Example configs: rich vs lazy (no gain/LR scan, just baseline)
    rich_cfg = (10, 12)    # narrow, deep
    lazy_cfg = (250, 2)    # wide, shallow

    baseline_gain  = 1.0
    baseline_lr    = 0.10

    print("\n=== RICH CONFIG (baseline gain & lr) ===")
    rich_stats = stats_for_config(
        rich_cfg[0], rich_cfg[1],
        gain=baseline_gain, base_lr=baseline_lr,
        seeds=seeds
    )

    print("\n=== LAZY CONFIG (baseline gain & lr) ===")
    lazy_stats = stats_for_config(
        lazy_cfg[0], lazy_cfg[1],
        gain=baseline_gain, base_lr=baseline_lr,
        seeds=seeds
    )

    print("\n---------- summary ----------")
    print(f"RICH  (N={rich_cfg[0]}, L={rich_cfg[1]})")
    print(f"  G_lambda      = {rich_stats['G_lambda']:.3e}")
    print(f"  G_J           = {rich_stats['G_J']:.3e}")
    print(f"  ρ(λ, margin)  = {rich_stats['rho_lambda_margin']:.3f}")
    print(f"  ρ(||J||,margin) = {rich_stats['rho_J_margin']:.3f}")

    print(f"\nLAZY  (N={lazy_cfg[0]}, L={lazy_cfg[1]})")
    print(f"  G_lambda      = {lazy_stats['G_lambda']:.3e}")
    print(f"  G_J           = {lazy_stats['G_J']:.3e}")
    print(f"  ρ(λ, margin)  = {lazy_stats['rho_lambda_margin']:.3f}")
    print(f"  ρ(||J||,margin) = {lazy_stats['rho_J_margin']:.3f}")

    # ----------------------------------------------------------------
    # OPTIONAL: Uncomment this to run the full (gain, lr, depth, width) grid.
    # WARNING: this is very expensive (PGD + FTLE per config).
    # ----------------------------------------------------------------
    G_lam_grid, G_J_grid, rho_lam_grid, rho_J_grid = run_full_grid(
        WIDTHS, DEPTHS, GAINS, BASE_LRS, seeds=seeds, eps_hi=0.30, grid=161
    )
    print(WIDTHS, DEPTHS, GAINS, BASE_LRS)
    np.save("G_lambda_grid.npy",   G_lam_grid)
    np.save("G_J_grid.npy",        G_J_grid)
    np.save("rho_lambda_grid.npy", rho_lam_grid)
    np.save("rho_J_grid.npy",      rho_J_grid)
