# ---------------------------------------------------------------
#  figure2_reproduce.py  –  *with progress bars*
#
#  Reproduces Fig. 2 of
#     L. Storm et al., “Finite-Time Lyapunov Exponents of Deep
#     Neural Networks”, *Phys. Rev. Lett.* 132 (2024) 057301.
#  Experimental protocol follows the Supplemental Material.
#
#  Only two small changes versus the previous version:
#    • added `tqdm` for progress bars during training and
#      finite-time Lyapunov–field evaluation;
#    • imported `tqdm` at the top.
#
#  Everything else is byte-for-byte identical.
# ---------------------------------------------------------------
import math, itertools, random, argparse, functools, pathlib, os

import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd.functional import jacobian
import matplotlib.pyplot as plt
from matplotlib import ticker
from tqdm import tqdm                    # ← NEW

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# ------------- reproducibility ------------------------------------------------
SEED = 20240201
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_default_dtype(torch.float32)

# ------------- hyper-parameters (faithful to Storm 2024) -----------------------
N_LIST = [10, 50, 250]         # widths (rows)
L_LIST = [2, 4, 8, 12]         # depths (columns)

TOTAL_SAMPLES   = 40_000
TRAIN_FRACTION  = 0.90
BATCH_SIZE      = 8192
EPOCHS          = 400
LR              = 0.05
GRID_RES        = 201
QUIVER_STEP     = 6
DEVICE          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ------------- dataset --------------------------------------------------------
def make_circular_dataset(n: int, side: float = 2.0):
    r = math.sqrt(0.5 * side**2 / math.pi)               # 50 % split
    xy = torch.empty(n, 2).uniform_(-side/2, side/2)
    radius = torch.linalg.vector_norm(xy, dim=1)
    t = torch.where(radius <= r, -torch.ones_like(radius), torch.ones_like(radius))
    return xy, t.unsqueeze(1)

def standardise(train_x, *arrays):
    μ = train_x.mean(0, keepdim=True)
    σ = train_x.std(0, unbiased=False, keepdim=True)
    return [(a - μ) / σ for a in (train_x,) + arrays]

# ------------- network --------------------------------------------------------
class TanhMLP(nn.Module):
    def __init__(self, width: int, depth: int):
        super().__init__()
        layers, in_dim = [], 2
        for _ in range(depth):
            layers += [nn.Linear(in_dim, width, bias=True), nn.Tanh()]
            in_dim = width
        self.feature_extractor = nn.Sequential(*layers)
        self.readout = nn.Linear(in_dim, 1, bias=True)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 1.0 / math.sqrt(m.weight.shape[1]))
                nn.init.zeros_(m.bias)

    def forward(self, x, return_hidden: bool = False):
        h = self.feature_extractor(x)
        return h if return_hidden else self.readout(h)

# ------------- training -------------------------------------------------------
def train(model, loader, desc: str = "training"):
    opt, mse = torch.optim.SGD(model.parameters(), lr=LR), nn.MSELoss()
    model.train()
    for _ in tqdm(range(EPOCHS), desc=desc, leave=False):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            mse(model(xb), yb).backward()
            opt.step()

# ------------- FTLE / stretching direction -----------------------------------
def jac_hidden(model, x):
    x = x.detach().requires_grad_(True)
    h = lambda inp: model(inp, return_hidden=True)
    J = jacobian(h, x, vectorize=False)
    return J.squeeze()                   # (width, 2)

def ftle_and_dir(model, grid, depth, desc="FTLE grid"):
    M = grid.shape[0]
    λ_field  = np.empty((M, M), dtype=np.float32)
    v_field  = np.empty((M, M, 2), dtype=np.float32)

    for i in tqdm(range(M), desc=desc, leave=False):
        for j in range(M):
            x = torch.tensor(grid[i, j], device=DEVICE, dtype=torch.float32)
            J = jac_hidden(model, x)
            _, S, Vh = torch.linalg.svd(J, full_matrices=False)
            σ_max = S[0].clamp(min=1e-12)
            λ1    = math.log(float(σ_max)) / depth
            λ_field[i, j] = depth * λ1
            v_field[i, j] = Vh[0].cpu().numpy()
    return λ_field, v_field

# ------------- utilities ------------------------------------------------------
def make_grid(res: int):
    lin = np.linspace(-math.sqrt(3), math.sqrt(3), res, dtype=np.float32)
    xx, yy = np.meshgrid(lin, lin, copy=False)
    return np.stack((xx, yy), axis=2).astype(np.float32, copy=False)

def accuracy(model, loader):
    model.eval()
    hits, total = 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            hits += (torch.sign(model(xb)) == yb).sum().item()
            total += yb.numel()
    return hits / total

# ------------- main -----------------------------------------------------------
def main(outpath: str = "figure2.png"):
    x, t = make_circular_dataset(TOTAL_SAMPLES)
    idx  = torch.randperm(TOTAL_SAMPLES)
    tr_n = int(TRAIN_FRACTION * TOTAL_SAMPLES)
    x_tr, t_tr, x_te, t_te = x[idx[:tr_n]], t[idx[:tr_n]], x[idx[tr_n:]], t[idx[tr_n:]]

    x_tr, x_te = standardise(x_tr, x_te)

    train_ds = TensorDataset(x_tr, t_tr)
    test_ds  = TensorDataset(x_te, t_te)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_ds , batch_size=8192, shuffle=False)

    fig, axes = plt.subplots(len(N_LIST), len(L_LIST),
                             figsize=(12, 8), sharex=True, sharey=True,
                             subplot_kw={"aspect": 1.0})
    cmap, grid = plt.get_cmap("coolwarm"), make_grid(GRID_RES)
    extent, vmax_global = [-math.sqrt(3), math.sqrt(3)]*2, 0.0

    for row, N in enumerate(N_LIST):
        for col, L in enumerate(L_LIST):
            ax = axes[row, col]
            model = TanhMLP(width=N, depth=L).to(DEVICE)

            train(model, train_loader, desc=f"Train N={N},L={L}")
            acc = accuracy(model, test_loader)
            print(acc)
            #assert acc >= 0.95, f"accuracy {acc:.3f} < 0.98 (N={N}, L={L})"

            λ, v = ftle_and_dir(model, grid, depth=L,
                                desc=f"FTLE N={N},L={L}")
            vmax_global = max(vmax_global, λ.max(), -λ.min())

            im = ax.imshow(λ, origin="lower", extent=extent,
                           vmin=-vmax_global, vmax=vmax_global, cmap=cmap)
            qs = slice(None, None, QUIVER_STEP)
            ax.quiver(grid[qs, qs, 0], grid[qs, qs, 1],
                      v[qs, qs, 0], v[qs, qs, 1],
                      color="k", pivot="mid", scale=30, headwidth=2)

            if row == 0:  ax.set_title(f"L = {L}", fontsize=11)
            if col == 0:  ax.set_ylabel(f"N = {N}", fontsize=11,
                                        rotation=0, labelpad=25)
            ax.set_xticks([]);  ax.set_yticks([])

    cax = fig.add_axes([0.92, 0.15, 0.025, 0.7])
    norm = plt.Normalize(vmin=-vmax_global, vmax=vmax_global)
    cb   = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                        cax=cax)
    cb.set_label(r"$L\,\lambda^{(L)}_1(x)$", fontsize=12)
    cb.ax.yaxis.set_major_locator(ticker.MaxNLocator(5))

    plt.subplots_adjust(wspace=0.02, hspace=0.02, right=0.9)
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    print(f"Saved {outpath}")

# -----------------------------------------------------------------------------    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reproduce Figure 2 of Storm et al. (2024)")
    parser.add_argument("--output", default="figure2.png",
                        help="output PNG path (default: figure2.png)")
    args = parser.parse_args()
    main(args.output)
