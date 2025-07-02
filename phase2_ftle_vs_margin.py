# phase2_ftle_vs_margin.py
"""
Phase 2:  FTLE  ↔  Adversarial Margin
------------------------------------

* Loads the rich (N = 10, L = 12) and lazy (N = 250, L = 2) checkpoints that
  Phase-1 already trained to ≥ 95 % training accuracy.
* For every test point it finds the *smallest* PGD perturbation ε*
  that flips the label (10-step logarithmic bisection, ε∈[0,0.30]).
* Looks up the pre-computed FTLE value λ₁(x) on a 161×161 grid.
* Computes Spearman ρ(λ₁, ε*) and plots quartile AUC curves.

"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import os, math, random, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
#  <<<  IMPORT objects from Phase-1  >>>
# ---------------------------------------------------------------------
from phase_1_layers_representational_alignment import (
    FC, make_circle, train_to_acc         # same file / same folder
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------
#  1.  load checkpoints (already ≥95 % accurate)
# ---------------------------------------------------------------------
CHK_DIR = "models"        # Phase-1 saved .pt files here
FTLE_DIR = "ftle"         # we will cache FTLE grids here
os.makedirs(CHK_DIR,  exist_ok=True)
os.makedirs(FTLE_DIR, exist_ok=True)

def load_net(width, depth, seed):
    net = FC(width, depth).to(device)
    path = f"{CHK_DIR}/net_w{width}_L{depth}_seed{seed}.pt"
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found.  Run phase-1 first to create checkpoints.")
    net.load_state_dict(torch.load(path, map_location=device))
    net.eval()
    return net

# ---------------------------------------------------------------------
#  2.  FTLE grid helper (161² cached per seed)
# ---------------------------------------------------------------------
def ftle_field(net, depth, grid=161, bbox=(-1.2, 1.2)):
    xs = torch.linspace(*bbox, grid, device=device)
    ys = torch.linspace(*bbox, grid, device=device)
    field = np.empty((grid, grid), np.float32)
    with torch.no_grad():
        for i, xv in enumerate(xs):
            for j, yv in enumerate(ys):
                x = torch.tensor([xv, yv], requires_grad=True, device=device)
                J = torch.autograd.functional.jacobian(
                    lambda z: net(z.unsqueeze(0), hid=True).squeeze(0), x)
                sigmax = torch.linalg.svdvals(J).max()
                field[j, i] = (1 / depth) * torch.log(sigmax).item()
    return field  # numpy

def load_ftle_grid(width, depth, seed, grid=161):
    path = f"{FTLE_DIR}/ftle_w{width}_L{depth}_seed{seed}.npy"
    if os.path.exists(path):
        return np.load(path)
    net = load_net(width, depth, seed)
    fld = ftle_field(net, depth, grid)
    np.save(path, fld)
    return fld

def ftle_lookup(fld, x, bbox=(-1.2, 1.2)):
    gx, gy = fld.shape[1]-1, fld.shape[0]-1
    xmin, xmax = bbox
    i = int((x[0]-xmin)/(xmax-xmin)*gx)
    j = int((x[1]-xmin)/(xmax-xmin)*gy)
    i = np.clip(i, 0, gx); j = np.clip(j, 0, gy)
    return fld[j, i]

# ---------------------------------------------------------------------
#  3.  PGD attack  &  margin search
# ---------------------------------------------------------------------
def pgd(net, x, y, eps, step, k=20):
    delta = torch.zeros_like(x, requires_grad=True)
    for _ in range(k):
        out = net(x + delta)
        loss = - (y * out).mean()
        loss.backward()
        delta.data = (delta + step * delta.grad.sign()).clamp(-eps, eps)
        delta.grad.zero_()
    return (x + delta).detach()

def margin(net, x, y, eps_hi=0.30):
    eps_lo = 0.0
    for _ in range(10):                     # bisection depth → ~1e-3
        mid = 0.5 * (eps_lo + eps_hi)
        adv = pgd(net, x, y, mid, step=mid/10)
        if torch.sign(net(adv)) != y:
            eps_hi = mid
        else:
            eps_lo = mid
    return eps_hi

# ---------------------------------------------------------------------
#  4.  main experiment for one config
# ---------------------------------------------------------------------
def evaluate_model(width, depth, seeds, eps_hi=0.30, grid=161):
    (xt, yt), (xe, ye) = make_circle()
    test_loader = DataLoader(TensorDataset(xe, ye),
                             batch_size=1024, shuffle=False)
    all_margins, all_ftle = [], []
    for seed in seeds:
        net  = load_net(width, depth, seed)
        fld  = load_ftle_grid(width, depth, seed, grid)
        for xb, yb in tqdm(test_loader,
                           desc=f"seed {seed}  N={width} L={depth}"):
            xb, yb = xb.to(device), yb.to(device)
            for xi, yi in zip(xb, yb):
                eps_star = margin(net, xi.unsqueeze(0), yi, eps_hi)
                all_margins.append(eps_star)
                all_ftle.append(ftle_lookup(fld, xi.cpu().numpy()))
    return np.array(all_margins), np.array(all_ftle)

# ---------------------------------------------------------------------
#  5.  Spearman ρ  &  plots
# ---------------------------------------------------------------------
def summary(width, depth, seeds=range(9)):
    m, f = evaluate_model(width, depth, seeds)
    rho, p = spearmanr(f, m)
    print(f"N={width:3} L={depth:<3}  Spearman ρ = {rho:.3f}  (p={p:.1e})")
    # density scatter
    plt.figure(figsize=(4,4))
    plt.hexbin(f, m, gridsize=60, cmap="magma", mincnt=1)
    plt.xlabel("λ₁ (FTLE)"); plt.ylabel("margin ε*")
    plt.title(f"Rich={width==10}")
    plt.colorbar(label="count")
    plt.tight_layout()
    plt.savefig(f"plots/ftle_vs_margin_N{width}_L{depth}.png", dpi=300)
    # quartile AUC curves
    q = np.quantile(f, [0.25,0.5,0.75])
    idx_q = np.digitize(f, q)            # 0,1,2,3
    eps_grid = np.arange(0.0, 0.31, 0.02)
    for qbin in range(4):
        succ = [(m[idx_q==qbin] <= eps).mean() for eps in eps_grid]
        plt.plot(eps_grid, succ, label=f"Q{qbin+1}")
    plt.legend(); plt.xlabel("ε"); plt.ylabel("attack success")
    plt.title(f"AUC by FTLE quartile  (N={width},L={depth})")
    plt.tight_layout()
    plt.savefig(f"plots/auc_quartiles_N{width}_L{depth}.png", dpi=300)
    return rho

# ---------------------------------------------------------------------
if __name__ == "__main__":
    torch.set_grad_enabled(True)
    os.makedirs("plots", exist_ok=True)

    seeds = list(range(9))
    rho_rich = summary(10, 12, seeds)
    rho_lazy = summary(250, 2, seeds)

    print("\n---------- summary ----------")
    print(f"rich (10×12)  ρ = {rho_rich:.3f}")
    print(f"lazy (250×2)  ρ = {rho_lazy:.3f}")
    if rho_rich <= -0.65:
        print("✓ Strong inverse correlation for rich net.")
    if rho_lazy <= -0.65:
        print("✓ Strong inverse correlation for lazy net.")
