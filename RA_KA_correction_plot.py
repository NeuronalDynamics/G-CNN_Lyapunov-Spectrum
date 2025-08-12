# ra_ka_corrections_plot.py
# Grid sweep with the corrected RA/KA protocol + C-3/C-4 plots

import os, math, random, copy, numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tqdm import tqdm

# ===================== CONFIG =====================
WIDTHS            = [50, 100, 150, 200, 250, 300]
DEPTHS            = [2, 4, 8, 10, 12]
TRIALS_PER_CELL   = 3
BATCH_SIZE        = 8192

TARGET_ACC        = 0.95
MAX_EPOCHS        = 2000
BASE_LR           = 0.05

# Toggle depth factor: True => η = base_lr/(fan_in * depth), False => base_lr/fan_in
USE_DEPTH_FACTOR  = True

# Training budget style
TRAINING_MODE     = "early_stop"     # "early_stop" or "fixed_steps"
FIXED_STEPS       = 5000             # if using fixed_steps

# KA is slower; toggle if needed
COMPUTE_KA        = True
KA_SUBSET         = 256

# Caching trained checkpoints so we don't retrain
CACHE_DIR         = "results_corr_plots"
os.makedirs(CACHE_DIR, exist_ok=True)
# ==================================================

# --------------------------- data ---------------------------
TOTAL_PTS, TRAIN_SPLIT = 40_000, 0.9
def make_circle():
    xy = np.random.uniform(-1, 1, (TOTAL_PTS, 2))
    r2 = (xy**2).sum(1)
    thr = np.median(r2)
    y   = ((r2 < thr).astype(np.float32)*2 - 1)
    idx = np.random.permutation(TOTAL_PTS)
    tr  = int(TRAIN_SPLIT*TOTAL_PTS)
    to_t= lambda a: torch.tensor(a, dtype=torch.float32)
    return (to_t(xy[idx[:tr]]),  to_t(y[idx[:tr]])[:,None]), \
           (to_t(xy[idx[tr:]]), to_t(y[idx[tr:]])[:,None])

# --------------------------- model ---------------------------
class FC(nn.Module):
    def __init__(self, width:int, depth:int):
        super().__init__()
        self.depth = depth
        self.hid = nn.ModuleList()
        prev = 2
        for _ in range(depth):
            l = nn.Linear(prev, width)
            nn.init.normal_(l.weight, 0., 1/math.sqrt(prev))
            nn.init.zeros_(l.bias)
            self.hid.append(l); prev = width
        self.out = nn.Linear(prev, 1)
        nn.init.normal_(self.out.weight, 0., 1/math.sqrt(prev))
        nn.init.zeros_(self.out.bias)

    def forward(self, x, *, hid=False, grad=False):
        for l in self.hid: x = torch.tanh(l(x))
        if hid:  return x
        if grad: return self.out(x)     # logits for Jacobian/grad
        return torch.tanh(self.out(x))

# --------------------- μP/NTK optimiser ---------------------
def per_layer_lr(layer: nn.Linear, base_lr: float, depth: int, use_depth: bool):
    fan_in = layer.weight.data.size(1)
    denom  = fan_in * (depth if use_depth else 1)
    return base_lr / denom

def make_optim(net: FC, base_lr: float, use_depth: bool):
    groups = []
    for m in net.modules():
        if isinstance(m, nn.Linear):
            lr = per_layer_lr(m, base_lr, net.depth, use_depth)
            groups.append({"params": [m.weight], "lr": lr})
            if m.bias is not None:
                groups.append({"params": [m.bias], "lr": lr})
    return torch.optim.SGD(groups, momentum=0.0)

# -------------------------- helpers --------------------------
def loader_acc(net: FC, loader: DataLoader) -> float:
    net.eval(); ok=tot=0
    with torch.no_grad():
        for x, y in loader:
            ok  += (torch.sign(net(x)) == y).sum().item()
            tot += y.size(0)
    return ok/tot

def snapshot_params(net: FC):
    return {k: v.detach().clone() for k, v in net.state_dict().items()}

def drift_global(theta0: dict, net: FC) -> float:
    s = 0.0
    sd = net.state_dict()
    for k in theta0.keys():
        s += (sd[k] - theta0[k]).pow(2).sum().item()
    return math.sqrt(s)

def drift_per_layer(theta0: dict, net: FC):
    sd = net.state_dict()
    sums = {}
    for name, p0 in theta0.items():
        if name.startswith("hid."):
            idx = int(name.split('.')[1])
            sums.setdefault(f"hid.{idx}", 0.0)
            sums[f"hid.{idx}"] += (sd[name] - p0).pow(2).sum().item()
        elif name.startswith("out."):
            sums.setdefault("out", 0.0)
            sums["out"] += (sd[name] - p0).pow(2).sum().item()
    return {k: math.sqrt(v) for k,v in sums.items()}

def print_lr_summary(width, depth, base_lr, use_depth):
    lr_first  = base_lr / (2 * (depth if use_depth else 1))
    lr_hidden = base_lr / (width * (depth if use_depth else 1))
    print(f"    [lr] depth_factor={use_depth}  lr_first={lr_first:.3e}  lr_hidden/out={lr_hidden:.3e}")

# --------------- training (two modes + logs) ---------------
def train_with_logging(net: FC, train_loader: DataLoader,
                       base_lr=BASE_LR, max_epochs=MAX_EPOCHS,
                       mode=TRAINING_MODE):
    opt, mse = make_optim(net, base_lr, USE_DEPTH_FACTOR), nn.MSELoss()
    steps = 0
    theta0 = snapshot_params(net)
    print_lr_summary(net.hid[0].out_features, net.depth, base_lr, USE_DEPTH_FACTOR)

    def one_epoch():
        nonlocal steps
        net.train()
        for x, y in train_loader:
            opt.zero_grad(); mse(net(x), y).backward(); opt.step(); steps += 1

    epochs_done = 0
    if mode == "fixed_steps":
        target_steps = FIXED_STEPS
        while steps < target_steps:
            one_epoch()
            if steps % 500 == 0:
                acc = loader_acc(net, train_loader)
                print(f"    [train] steps={steps:5d} acc={acc:.3f}")
        epochs_done = None
    else:  # early_stop
        for ep in range(1, max_epochs+1):
            one_epoch(); epochs_done = ep
            if ep % 10 == 0:
                acc = loader_acc(net, train_loader)
                print(f"    [train] epoch={ep:4d} acc={acc:.3f}")
                if acc >= TARGET_ACC:
                    print(f"    [early-stop] hit {TARGET_ACC:.2f} at epoch {ep}")
                    break

    drift_g = drift_global(theta0, net)
    print(f"    [drift] global_L2={drift_g:.3e}  per_layer={drift_per_layer(theta0, net)}")
    return epochs_done, steps, drift_g

# ------------------------- RA & KA -------------------------
def gram(M): return M.T @ M
def frob(A): return torch.norm(A, p='fro')
def align(A,B): return torch.trace(A@B)/(frob(A)*frob(B)+1e-8)

def collect_acts_per_layer(net: FC, loader: DataLoader):
    bufs = [[] for _ in net.hid]
    hooks = [m.register_forward_hook(
             (lambda i: (lambda _m,_i,out: bufs[i].append(out.detach())))(i))
             for i, m in enumerate(net.hid)]
    with torch.no_grad():
        for xb, _ in loader: net(xb)
    [h.remove() for h in hooks]
    return [torch.cat(b, 0) for b in bufs]

def compute_RA(init_net: FC, trained_net: FC, probe_loader: DataLoader):
    acts0 = collect_acts_per_layer(init_net,    probe_loader)
    actsT = collect_acts_per_layer(trained_net, probe_loader)
    vals = [align(gram(HT), gram(H0)).item() for H0, HT in zip(acts0, actsT)]
    return float(np.mean(vals)), vals

def compute_KA(init_net: FC, trained_net: FC, loader: DataLoader, subset=KA_SUBSET):
    params0 = [p for p in init_net.parameters()    if p.requires_grad]
    paramsT = [p for p in trained_net.parameters() if p.requires_grad]
    assert len(params0) == len(paramsT)
    cuts = np.cumsum([0] + [p.numel() for p in params0])

    xb, _ = next(iter(DataLoader(loader.dataset, batch_size=subset, shuffle=True)))

    def flat_grad(net, params, x):
        net.zero_grad()
        y = net(x.unsqueeze(0), grad=True)
        grads = torch.autograd.grad(y, params, retain_graph=True, allow_unused=True)
        flat = []
        for g, p in zip(grads, params):
            flat.append(g.reshape(-1) if g is not None else torch.zeros_like(p).reshape(-1))
        return torch.cat(flat)

    G0 = torch.stack([flat_grad(init_net,    params0, x) for x in xb])
    GT = torch.stack([flat_grad(trained_net, paramsT, x) for x in xb])

    KA_vals = []
    for i in range(len(cuts)-1):
        s, e = cuts[i], cuts[i+1]
        K0 = G0[:, s:e] @ G0[:, s:e].T
        KT = GT[:, s:e] @ GT[:, s:e].T
        KA_vals.append(align(KT, K0).item())
    return float(np.mean(KA_vals)), KA_vals

# ----------------------- checkpointing -----------------------
def ckpt_path(N, L, seed):
    return os.path.join(CACHE_DIR, f"model_N{N}_L{L}_seed{seed}.pt")

def train_new(N, L, seed, train_loader):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    net = FC(N, L)
    print(f"[train] N={N:<4} L={L:<3} seed={seed}  mode={TRAINING_MODE}  use_depth_factor={USE_DEPTH_FACTOR}")
    epochs, steps, drift_g = train_with_logging(net, train_loader)
    torch.save({'state_dict': net.state_dict(),
                'meta': {'base_lr': BASE_LR,
                         'use_depth_factor': USE_DEPTH_FACTOR,
                         'training_mode': TRAINING_MODE,
                         'epochs': epochs, 'steps': steps,
                         'drift_global': drift_g}},
               ckpt_path(N, L, seed))
    return net

def load_or_train(N, L, seed, train_loader):
    path = ckpt_path(N, L, seed)
    if os.path.exists(path):
        data = torch.load(path, map_location='cpu')
        meta = data.get('meta', {})
        net  = FC(N, L); net.load_state_dict(data['state_dict'])
        acc = loader_acc(net, train_loader)
        print(f"[ckpt]  load {os.path.basename(path)}  acc={acc:.3f}  meta={meta}")
        # retrain if settings mismatch or under target
        if acc < TARGET_ACC or meta.get('use_depth_factor') != USE_DEPTH_FACTOR \
           or meta.get('training_mode') != TRAINING_MODE \
           or not math.isclose(meta.get('base_lr', BASE_LR), BASE_LR):
            print("        → retraining to match current settings")
            return train_new(N, L, seed, train_loader)
        return net
    else:
        return train_new(N, L, seed, train_loader)

# --------------------------- grid run ---------------------------
def run_grid():
    (xt, yt), (xe, ye) = make_circle()
    train_loader = DataLoader(TensorDataset(xt, yt), batch_size=BATCH_SIZE, shuffle=True)
    probe_loader = DataLoader(TensorDataset(xe, ye), batch_size=BATCH_SIZE, shuffle=False)

    RA_map = np.zeros((len(DEPTHS), len(WIDTHS)))
    KA_map = np.zeros_like(RA_map)

    for di, L in enumerate(DEPTHS):
        for wi, N in enumerate(WIDTHS):
            ra_vals, ka_vals = [], []
            for seed in range(TRIALS_PER_CELL):
                # init_net must match the trained seed init
                torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
                init_net = FC(N, L)
                trained  = load_or_train(N, L, seed, train_loader)

                mean_ra, layer_ra = compute_RA(init_net, trained, probe_loader)
                print(f"    [RA]  N={N:<4} L={L:<3} seed={seed}  mean={mean_ra:.3f}  per_layer={np.round(layer_ra,3)}")
                ra_vals.append(mean_ra)

                if COMPUTE_KA:
                    mean_ka, layer_ka = compute_KA(init_net, trained, train_loader)
                    print(f"    [KA]  N={N:<4} L={L:<3} seed={seed}  mean={mean_ka:.3f}  (per-layer count={len(layer_ka)})")
                    ka_vals.append(mean_ka)

            RA_map[di, wi] = float(np.mean(ra_vals))
            if COMPUTE_KA:
                KA_map[di, wi] = float(np.mean(ka_vals))
            print(f"[cell] N={N:<4} L={L:<3}  RA={RA_map[di,wi]:.3f}  " +
                  (f"KA={KA_map[di,wi]:.3f}" if COMPUTE_KA else ""))

    return RA_map, KA_map

# --------------------------- plotting ---------------------------
def plot_c3_c4(RA, KA, widths, depths, title_suffix=""):
    # RA, KA: shape [len(depths), len(widths)]
    RA_T, KA_T = RA.T, KA.T  # shape [len(widths), len(depths)]
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # C-3: RA vs depth (one curve per width)
    for wi, N in enumerate(widths):
        ax[0].plot(depths, RA_T[wi], 'o-', label=f"N={N}")
    ax[0].set_title("C-3  RA vs depth" + title_suffix)
    ax[0].set_xlabel("Depth L"); ax[0].set_ylabel("RA"); ax[0].set_ylim(0, 1); ax[0].grid(True); ax[0].legend()

    # C-4: KA vs depth (one curve per width)
    for wi, N in enumerate(widths):
        ax[1].plot(depths, KA_T[wi], 's--', label=f"N={N}")
    ax[1].set_title("C-4  KA vs depth" + title_suffix)
    ax[1].set_xlabel("Depth L"); ax[1].set_ylabel("KA"); ax[1].set_ylim(0, 1); ax[1].grid(True); ax[1].legend()

    plt.tight_layout(); plt.show()

# ----------------------------- main -----------------------------
if __name__ == "__main__":
    torch.set_grad_enabled(True)
    RA, KA = run_grid()

    print("\n==== SUMMARY (avg over seeds) ====")
    for di, L in enumerate(DEPTHS):
        for wi, N in enumerate(WIDTHS):
            if COMPUTE_KA:
                print(f"N={N:<4} L={L:<3}  RA={RA[di,wi]:.3f}  KA={KA[di,wi]:.3f}")
            else:
                print(f"N={N:<4} L={L:<3}  RA={RA[di,wi]:.3f}")

    suffix = f"  (depth_factor={USE_DEPTH_FACTOR}, mode={TRAINING_MODE})"
    plot_c3_c4(RA, KA, WIDTHS, DEPTHS, title_suffix=suffix)
