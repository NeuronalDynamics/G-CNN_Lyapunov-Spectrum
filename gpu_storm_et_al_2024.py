"""
Reproduce Figure 2 and Figure 3 from
    L. Storm, H. Linander, J. Bec, K. Gustavsson & B. Mehlig (2024)
    “Finite-Time Lyapunov Exponents of Deep Neural Networks”

2025-06-06  – GPU edition (+ tqdm progress bars)
------------------------------------------------
• Train/evaluate on GPU if available.
• All tensors & models are sent to `device`, defined once at the top.
• Everything else unchanged from the progress-bar version.
"""
# =====================================================================
#                               SET-UP
# =====================================================================
from __future__ import annotations
import math, random
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"   # ← allow duplicate runtimes

# ---------------------------  CONFIG  --------------------------------
SEED        = 2024
TOTAL_PTS   = 40_000
TRAIN_SPLIT = 0.90
GRID_RES    = 161

EPOCHS      = 200
BATCH_SIZE  = 8192
LR          = 0.05

WIDTHS_F2   = [10, 50, 250]
DEPTHS_F2   = [2, 4, 8, 12]
DEPTHS_F3   = [2, 4, 8, 12]
WIDTHS_F3   = np.logspace(0, 2.6, 14, dtype=int)   # ≈ 10 … 400

# ---------------------------  DEVICE  --------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO]  using device: {device}")

# ---------------------  reproducibility  -----------------------------
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# =====================================================================
#                           DATA GENERATION
# =====================================================================
def make_circle_dataset(n_pts: int = TOTAL_PTS):
    xy = np.random.uniform(-1, 1, size=(n_pts, 2))
    r2 = (xy**2).sum(1)
    median = np.median(r2)
    y = np.where(r2 < median, -1.0, 1.0)

    shuf = np.random.permutation(n_pts)
    split = int(TRAIN_SPLIT * n_pts)
    tr, te = shuf[:split], shuf[split:]
    x_tr, x_te = xy[tr], xy[te]
    y_tr, y_te = y[tr], y[te]

    μ, σ = x_tr.mean(0, keepdims=True), x_tr.std(0, keepdims=True)
    x_tr = (x_tr - μ) / σ
    x_te = (x_te - μ) / σ

    to_t = lambda a: torch.tensor(a, dtype=torch.float32)
    y_uns = lambda a: to_t(a).unsqueeze(1)
    return (to_t(x_tr), y_uns(y_tr)), (to_t(x_te), y_uns(y_te))

# =====================================================================
#                           MODEL
# =====================================================================
class FCNet(nn.Module):
    def __init__(self, in_dim: int, width: int, depth: int, out_dim: int = 1):
        super().__init__()
        self.hidden = nn.ModuleList()
        prev = in_dim
        for _ in range(depth):
            l = nn.Linear(prev, width)
            nn.init.normal_(l.weight, 0.0, 1/math.sqrt(prev))
            nn.init.zeros_(l.bias)
            self.hidden.append(l)
            prev = width
        self.out = nn.Linear(prev, out_dim)
        nn.init.normal_(self.out.weight, 0.0, 1/math.sqrt(prev))
        nn.init.zeros_(self.out.bias)

    def forward(self, x, *, return_hidden=False):
        for l in self.hidden:
            x = torch.tanh(l(x))
        if return_hidden:
            return x
        return torch.tanh(self.out(x))     # labels are ±1

# =====================================================================
#                           TRAINING
# =====================================================================
def train(model: FCNet, loader: DataLoader, *, desc="train"):
    model.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=LR)
    mse = nn.MSELoss()
    model.train()
    for _ in tqdm(range(EPOCHS), desc=desc, leave=False):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = mse(model(xb), yb)
            loss.backward()
            opt.step()

# =====================================================================
#                   FINITE-TIME LYAPUNOV EXPONENT
# =====================================================================
def _max_sv(jac: torch.Tensor) -> torch.Tensor:
    return torch.linalg.svdvals(jac).max() if jac.numel() else torch.tensor(0., device=device)

def ftle_field(model: FCNet, depth: int, grid: int = GRID_RES):
    model.eval()
    xs = torch.linspace(-2, 2, grid, device=device)
    ys = torch.linspace(-2, 2, grid, device=device)
    field = np.empty((grid, grid), np.float32)
    vecs  = np.empty((grid, grid, 2), np.float32)

    with torch.no_grad():
        for i, xv in enumerate(tqdm(xs, desc="FTLE grid-x", leave=False)):
            for j, yv in enumerate(ys):
                x = torch.tensor([xv, yv], requires_grad=True, device=device)
                jac = torch.autograd.functional.jacobian(
                    lambda z: model(z.unsqueeze(0), return_hidden=True).squeeze(0), x)
                U, S, Vt = torch.linalg.svd(jac)
                k = torch.argmax(S)
                field[j, i] = (1/depth) * torch.log(S[k]).item()
                vecs[j, i]  = Vt[k].detach().cpu().numpy()
    return field, vecs

# =====================================================================
#                         FIGURE 2
# =====================================================================
def reproduce_figure2(data):
    (x_tr, y_tr), _ = data
    loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)

    import matplotlib
    matplotlib.use("Agg")   # avoids backend problems on headless servers
    fig, axes = plt.subplots(len(WIDTHS_F2), len(DEPTHS_F2),
                             figsize=(13, 8), sharex=True, sharey=True)
    total = len(WIDTHS_F2)*len(DEPTHS_F2)
    with tqdm(total=total, desc="Fig 2 networks") as bar:
        for r, N in enumerate(WIDTHS_F2):
            for c, L in enumerate(DEPTHS_F2):
                net = FCNet(2, N, L)
                train(net, loader, desc=f"L={L} N={N}")
                f, v = ftle_field(net, L)
                ax = axes[r, c]
                im = ax.imshow(L*f, extent=[-2,2,-2,2], origin="lower",
                               cmap="coolwarm", vmin=-4, vmax=4)
                step = max(1, GRID_RES//17)
                ys, xs = np.mgrid[0:GRID_RES:step, 0:GRID_RES:step]
                ax.quiver(np.linspace(-2,2,GRID_RES)[xs],
                          np.linspace(-2,2,GRID_RES)[ys],
                          v[ys,xs,0], v[ys,xs,1],
                          scale=20, width=0.002, color='k')
                if r==0: ax.set_title(f"L={L}")
                if c==0: ax.set_ylabel(f"N={N}")
                ax.set_xticks([]), ax.set_yticks([])
                bar.update(1)
    cbar = fig.colorbar(im, ax=axes, fraction=0.015)
    cbar.set_label(r"$L\,\lambda^{(L)}_1(x)$", rotation=270, labelpad=12)
    fig.suptitle("Reproduction of Figure 2 — FTLE ridges", y=0.93)
    fig.tight_layout()
    fig.savefig("figure2_ftle_ridges.png", dpi=300)

# =====================================================================
#                         FIGURE 3
# =====================================================================
def test_error(net: FCNet, data):
    _, (x_te, y_te) = data
    x_te, y_te = x_te.to(device), y_te.to(device)
    net.eval()
    with torch.no_grad():
        out = net(x_te)
        return (torch.sign(out) != torch.sign(y_te)).float().mean().item()

def reproduce_figure3(data):
    (x_tr, y_tr), _ = data
    loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)

    # ---------- panel (a): error vs width ---------------------------
    err_rand, err_full = [], []
    for N in tqdm(WIDTHS_F3, desc="Fig 3a width sweep"):
        net_r = FCNet(2, N, depth=2)
        for p in net_r.hidden.parameters(): p.requires_grad_(False)
        train(net_r, loader, desc=f"rand N={N}")
        err_rand.append(test_error(net_r, data))

        net_f = FCNet(2, N, depth=2)
        train(net_f, loader, desc=f"full N={N}")
        err_full.append(test_error(net_f, data))

    # ---------- panel (b): FTLE averages ----------------------------
    avg_d, avg_c = {L: [] for L in DEPTHS_F3}, {L: [] for L in DEPTHS_F3}
    for N in tqdm(WIDTHS_F3, desc="Fig 3b FTLE stats"):
        for L in DEPTHS_F3:
            net = FCNet(2, N, L)
            train(net, loader, desc=f"L={L} N={N}")
            fld, _ = ftle_field(net, L, grid=81)

            xs = np.linspace(-2, 2, 81)
            ys = np.linspace(-2, 2, 81)
            X, Y = np.meshgrid(xs, ys)
            r = np.sqrt(X**2 + Y**2)
            r_med = np.median(r)

            mask_d = np.isclose(r, r_med, atol=0.05)
            mask_c = r < 0.2*r_med
            avg_d[L].append(np.mean(fld[mask_d]))
            avg_c[L].append(np.mean(fld[mask_c]))

    # ------------------------ plotting ------------------------------
    import matplotlib
    matplotlib.use("Agg")
    fig = plt.figure(figsize=(12,4))

    # (a) -----------------------------------------------------------
    ax1 = fig.add_subplot(1,2,1)
    ax1.plot(WIDTHS_F3, np.array(err_rand)*100, "k-",  label="Random hidden, trained output")
    ax1.plot(WIDTHS_F3, np.array(err_full)*100, "k--", label="Fully trained")
    ax1.set_xscale("log")
    ax1.set_xlabel("Width N")
    ax1.set_ylabel("Test error (%)")
    ax1.legend(frameon=False)

    # (b) -----------------------------------------------------------
    ax2 = fig.add_subplot(1,2,2)
    colours = plt.cm.tab10(np.linspace(0,1,len(DEPTHS_F3)))
    for i, L in enumerate(DEPTHS_F3):
        ax2.plot(WIDTHS_F3, [L*v for v in avg_d[L]], colours[i], label=f"L={L} (boundary)")
        ax2.plot(WIDTHS_F3, [L*v for v in avg_c[L]], colours[i], ls="--", label=f"L={L} (centre)")
    ax2.set_xscale("log")
    ax2.set_xlabel("Width N")
    ax2.set_ylabel(r"$L\,\langle\lambda^{(L)}_1\rangle$")
    ax2.legend(frameon=False, ncol=2, fontsize=8)

    fig.suptitle("Reproduction of Figure 3", y=0.94)
    fig.tight_layout()
    fig.savefig("figure3_learning_regimes.png", dpi=300)

# =====================================================================
#                               MAIN
# =====================================================================
if __name__ == "__main__":
    data = make_circle_dataset()
    print("\n[+] Dataset built — starting reproductions …\n")
    reproduce_figure2(data)
    reproduce_figure3(data)
    print("\n[✓] All done.  Figures saved in current directory.\n")
