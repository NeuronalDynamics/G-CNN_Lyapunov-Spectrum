# ra_ka_best_method_accstop.py
# --------------------------------------------------------------
# Representation (RA) and Tangent-Kernel (KA) alignment trends
# Best-practice setup with *accuracy-based* early stopping.
#
# RA  = centered FeatureGram (linear CKA) on a fixed probe set.
# KA  = sample-NTK on a subset of the same probe set, aligned
#       pre vs post with Frobenius-normalized cosine.
# OPT = μP/NTK-style per-layer LR: eta ~ 1 / (fan_in).
# TRN = train until TRAIN_ACC_TARGET with checkpoint verify.
# --------------------------------------------------------------
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import os, math, random, copy, numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# -------------------- CONFIG (edit as needed) --------------------
DEVICE              = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WIDTHS              = [10, 50, 100, 150, 200]
DEPTHS              = [2, 6, 8, 10, 12, 14, 16]
TRIALS_PER_CELL     = 3
BATCH_SIZE_TRAIN    = 8192
BATCH_SIZE_PROBE    = 8192

# Optimizer scaling: eta = base_lr / fan_in
BASE_LR             = 0.10

# Training target & budget (now accuracy-based)
TRAIN_ACC_TARGET    = 0.95     # stop when train accuracy ≥ this
MAX_EPOCHS          = 4000
LOG_EVERY_EPOCHS    = 100      # reduce chatter

# KA compute budget
KA_SUBSET           = 64
SEED_BASE           = 0

# Plotting
DO_PLOTS            = True

# ======== SIMPLE CHECKPOINTING (verified on accuracy) ===============
CHECKPOINT_DIR = "rk_ckpts_v4"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def fmt_float(x: float) -> str:
    """
    Convert float to a compact, filename-safe string.
    Example: 0.1 -> '0p1', 1.25 -> '1p25', -0.5 -> 'm0p5'
    """
    s = f"{x:.3g}"
    s = s.replace('.', 'p')
    if s.startswith('-'):
        s = 'm' + s[1:]
    return s


def ckpt_path(N: int, L: int, gain: float, base_lr: float, seed: int) -> str:
    gstr  = fmt_float(gain)
    lrstr = fmt_float(base_lr)
    fname = f"model_N{N}_L{L}_g{gstr}_lr{lrstr}_seed{seed}.pt"
    return os.path.join(CHECKPOINT_DIR, fname)


# ------------------------- DATA -----------------------------------
TOTAL_PTS, TRAIN_SPLIT = 40_000, 0.9


def make_circle():
    xy = np.random.uniform(-1, 1, (TOTAL_PTS, 2))
    r2 = (xy**2).sum(1)
    thr = np.median(r2)
    y   = ((r2 < thr).astype(np.float32) * 2 - 1).astype(np.float32)
    idx = np.random.permutation(TOTAL_PTS)
    tr  = int(TRAIN_SPLIT * TOTAL_PTS)
    to_t = lambda a: torch.tensor(a, dtype=torch.float32)
    return (to_t(xy[idx[:tr]]),  to_t(y[idx[:tr]])[:, None]), \
           (to_t(xy[idx[tr:]]), to_t(y[idx[tr:]])[:, None])


# ------------------------- MODEL ----------------------------------
class FC(nn.Module):
    def __init__(self, width: int, depth: int, gain: float = 1.0):
        """
        Simple fully-connected tanh MLP with per-layer Gaussian init.

        gain controls the std of each layer:
            weight ~ N(0, gain / sqrt(fan_in))
        """
        super().__init__()
        self.depth = depth
        self.gain  = gain
        self.hid = nn.ModuleList()
        prev = 2
        for _ in range(depth):
            l = nn.Linear(prev, width)
            nn.init.normal_(l.weight, 0., gain / math.sqrt(prev))
            nn.init.zeros_(l.bias)
            self.hid.append(l)
            prev = width
        self.out = nn.Linear(prev, 1)
        nn.init.normal_(self.out.weight, 0., gain / math.sqrt(prev))
        nn.init.zeros_(self.out.bias)

    def forward(self, x, *, hid=False, grad=False):
        for l in self.hid:
            x = torch.tanh(l(x))
        if hid:
            return x
        if grad:
            # logits (no final tanh) for NTK Jacobians etc.
            return self.out(x)
        return torch.tanh(self.out(x))


# =================== LR SCALING (fan-in only) ===================

def per_layer_lr(layer: nn.Linear, base_lr: float) -> float:
    """μP/NTK-style: η ∝ 1 / fan_in."""
    fan_in = layer.weight.data.size(1)
    return base_lr / float(fan_in)


def make_optim(net: nn.Module, base_lr: float) -> torch.optim.Optimizer:
    groups = []
    for m in net.modules():
        if isinstance(m, nn.Linear):
            lr = per_layer_lr(m, base_lr)
            groups.append({"params": [m.weight], "lr": lr})
            if m.bias is not None:
                groups.append({"params": [m.bias], "lr": lr})
    return torch.optim.SGD(groups, momentum=0.0)


def print_lr_summary(net: nn.Module, base_lr: float):
    lrs = []
    for m in net.modules():
        if isinstance(m, nn.Linear):
            lrs.append(per_layer_lr(m, base_lr))
    if lrs:
        print(f"[lr] fan-in scaled; min={min(lrs):.3e}  max={max(lrs):.3e}  layers={len(lrs)}")


# -------------------------- TRAINING -------------------------------
def dataset_to_loader(pair, batch_size, shuffle, device=DEVICE):
    x, y = pair
    ds = TensorDataset(x.to(device), y.to(device))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def loader_acc_and_loss(net: FC, loader: DataLoader, mse: nn.Module):
    net.eval()
    correct = total = 0
    loss_sum = 0.0
    for xb, yb in loader:
        logits = net(xb)
        loss = mse(logits, yb)
        loss_sum += loss.item() * yb.size(0)
        correct += (torch.sign(logits) == yb).sum().item()
        total   += yb.size(0)
    return correct / total, loss_sum / total


def train_until_acc(net: FC,
                    train_loader: DataLoader,
                    acc_target: float,
                    max_epochs: int,
                    base_lr: float):
    opt = make_optim(net, base_lr)
    mse = nn.MSELoss()
    for ep in range(1, max_epochs + 1):
        net.train()
        for xb, yb in train_loader:
            opt.zero_grad()
            loss = mse(net(xb), yb)
            loss.backward()
            opt.step()
        if ep % LOG_EVERY_EPOCHS == 0 or ep == 1:
            acc, loss_eval = loader_acc_and_loss(net, train_loader, mse)
            print(f"[train] epoch={ep:4d}  acc={acc:.3f}  loss={loss_eval:.4f}")
        acc, _ = loader_acc_and_loss(net, train_loader, mse)
        if acc >= acc_target:
            print(f"[early-stop] hit acc_target={acc_target:.3f} at epoch {ep} (acc={acc:.3f})")
            break


def verify_or_train_checkpoint(N: int, L: int,
                               gain: float, base_lr: float,
                               seed: int,
                               train_loader: DataLoader,
                               acc_target: float,
                               max_epochs: int) -> nn.Module:
    """
    Load or train an FC(N,L,gain) with per-layer base_lr,
    using accuracy-based early stopping.
    """
    path = ckpt_path(N, L, gain, base_lr, seed)
    net = FC(N, L, gain=gain).to(DEVICE)
    mse = nn.MSELoss()

    if os.path.exists(path):
        state = torch.load(path, map_location=DEVICE)
        net.load_state_dict(state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state)
        acc, _ = loader_acc_and_loss(net, train_loader, mse)
        if acc >= acc_target:
            print(f"[ckpt] loaded {os.path.basename(path)}  (acc={acc:.3f} ≥ target)")
            net.eval()
            return net
        else:
            print(f"[ckpt] {os.path.basename(path)}  acc={acc:.3f} < target → continuing training")

    torch.manual_seed(SEED_BASE + seed); np.random.seed(SEED_BASE + seed); random.seed(SEED_BASE + seed)
    train_until_acc(net, train_loader, acc_target, max_epochs, base_lr)
    torch.save({"state_dict": net.state_dict()}, path)
    print(f"[ckpt] saved {os.path.basename(path)}")
    net.eval()
    return net


# ------------------------ ALIGNMENT UTILS --------------------------
def frob(A: torch.Tensor) -> torch.Tensor:
    return torch.norm(A, p='fro')


def align(A: torch.Tensor, B: torch.Tensor) -> float:
    return (torch.trace(A @ B) / (frob(A) * frob(B) + 1e-8)).item()


@torch.no_grad()
def sample_gram_centered(H: torch.Tensor) -> torch.Tensor:
    # H: [N, D] (samples × features)
    Hc = H - H.mean(dim=0, keepdim=True)
    return Hc @ Hc.T  # [N, N]


@torch.no_grad()
def representation_alignment(net_init: FC, net_trained: FC, probe_loader: DataLoader) -> float:
    acts0, actsT = [], []
    net_init.eval(); net_trained.eval()
    for xb, _ in probe_loader:
        acts0.append(net_init(xb, hid=True))
        actsT.append(net_trained(xb, hid=True))
    H0 = torch.cat(acts0, 0)
    HT = torch.cat(actsT, 0)
    K0 = sample_gram_centered(H0)
    KT = sample_gram_centered(HT)
    return align(KT, K0)


def ntk_alignment(net_init: FC, net_trained: FC,
                  probe_loader: DataLoader,
                  subset: int = KA_SUBSET) -> float:
    X_list = []
    with torch.no_grad():
        for xb, _ in probe_loader:
            X_list.append(xb)
    X = torch.cat(X_list, 0)
    Ns = min(subset, X.shape[0])
    idx = torch.randperm(X.shape[0], device=X.device)[:Ns]
    Xs  = X[idx]

    def grad_matrix(net: FC, xs: torch.Tensor) -> torch.Tensor:
        net.eval()
        params = [p for p in net.parameters() if p.requires_grad]
        rows = []
        for i in range(xs.shape[0]):
            net.zero_grad(set_to_none=True)
            y = net(xs[i:i+1], grad=True)  # logit
            y.sum().backward()
            flat = []
            for p in params:
                g = p.grad
                flat.append(g.reshape(-1) if g is not None else torch.zeros_like(p).reshape(-1))
            rows.append(torch.cat(flat))
        return torch.stack(rows, 0)

    G0 = grad_matrix(net_init,    Xs)
    GT = grad_matrix(net_trained, Xs)
    K0 = G0 @ G0.T
    KT = GT @ GT.T
    return align(KT, K0)


# --------------------------- RUNNERS -------------------------------
def run_cell(N, L, seeds, train_loader, probe_loader):
    """
    RA/KA for a single (N,L), using gain=1.0, base_lr=BASE_LR
    (so Phase-1 behaviour stays unchanged).
    """
    ra_vals, ka_vals = [], []
    for s in seeds:
        # initial net (pre-state) with the same seed
        torch.manual_seed(SEED_BASE + s); np.random.seed(SEED_BASE + s); random.seed(SEED_BASE + s)
        net_init = FC(N, L, gain=1.0).to(DEVICE)

        # verify/train checkpoint with fixed gain & base_lr
        net_tr = verify_or_train_checkpoint(
            N, L, gain=1.0, base_lr=BASE_LR, seed=s,
            train_loader=train_loader,
            acc_target=TRAIN_ACC_TARGET,
            max_epochs=MAX_EPOCHS
        )

        print_lr_summary(net_tr, BASE_LR)

        # RA & KA on the same probe set
        ra = representation_alignment(net_init, net_tr, probe_loader)
        ka = ntk_alignment(net_init, net_tr, probe_loader, subset=KA_SUBSET)
        ra_vals.append(ra); ka_vals.append(ka)

        del net_init, net_tr
        torch.cuda.empty_cache()

    return float(np.mean(ra_vals)), float(np.mean(ka_vals))


def run_grid(widths, depths):
    (xt, yt), (xe, ye) = make_circle()
    train_loader = dataset_to_loader((xt, yt), BATCH_SIZE_TRAIN, shuffle=True, device=DEVICE)
    probe_loader = dataset_to_loader((xe, ye), BATCH_SIZE_PROBE, shuffle=False, device=DEVICE)

    RA_map = np.zeros((len(depths), len(widths)), dtype=np.float32)
    KA_map = np.zeros_like(RA_map)

    for di, L in enumerate(depths):
        for wi, N in enumerate(widths):
            print(f"\n[cell] width={N} depth={L}")
            ra, ka = run_cell(N, L, seeds=range(TRIALS_PER_CELL),
                              train_loader=train_loader, probe_loader=probe_loader)
            RA_map[di, wi] = ra; KA_map[di, wi] = ka
            print(f"[cell-summary] N={N:<4} L={L:<3}  RA={ra:.3f}  KA={ka:.3f}")
    return RA_map, KA_map, widths, depths


# ---------------------------- PLOTTING ------------------------------
def plot_c3_c4(RA, KA, widths, depths, title_suffix=""):
    RA_T, KA_T = RA.T, KA.T
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for wi, N in enumerate(widths):
        ax[0].plot(depths, RA_T[wi], 'o-', label=f"N={N}")
    ax[0].set_title("C-3  RA vs depth" + title_suffix)
    ax[0].set_xlabel("Depth L"); ax[0].set_ylabel("RA"); ax[0].set_ylim(0, 1); ax[0].grid(True); ax[0].legend()
    for wi, N in enumerate(widths):
        ax[1].plot(depths, KA_T[wi], 's--', label=f"N={N}")
    ax[1].set_title("C-4  KA vs depth" + title_suffix)
    ax[1].set_xlabel("Depth L"); ax[1].set_ylabel("KA"); ax[1].set_ylim(0, 1); ax[1].grid(True); ax[1].legend()
    plt.tight_layout()
    plt.savefig('ra_ka_2.png')


# ------------------------------ MAIN -------------------------------
if __name__ == "__main__":
    torch.set_grad_enabled(True)
    RA, KA, widths, depths = run_grid(WIDTHS, DEPTHS)

    print("\n==== SUMMARY (avg over seeds) ====")
    for di, L in enumerate(depths):
        for wi, N in enumerate(widths):
            print(f"N={N:<4} L={L:<3}  RA={RA[di,wi]:.3f}  KA={KA[di,wi]:.3f}")

    if DO_PLOTS:
        plot_c3_c4(RA, KA, widths, depths)
