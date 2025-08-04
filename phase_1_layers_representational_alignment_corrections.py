import math, random, argparse, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange
import os

# -------------- checkpoint helper -----------------------------------
CHK_DIR = "models/correction"                 # top-level folder for weights
os.makedirs(CHK_DIR, exist_ok=True)

def save_ckpt(net, width, depth, seed):
    path = f"{CHK_DIR}/net_w{width}_L{depth}_seed{seed}.pt"
    torch.save(net.state_dict(), path)

# ----------------------------- data ---------------------------------
TOTAL_PTS, TRAIN_SPLIT = 40_000, 0.9
def make_circle():
    xy = np.random.uniform(-1, 1, (TOTAL_PTS, 2))
    y  = (np.sum(xy**2,1) < np.median(np.sum(xy**2,1))).astype(np.float32)*2-1
    idx = np.random.permutation(TOTAL_PTS)
    tr  = int(TRAIN_SPLIT*TOTAL_PTS)
    xt, yt = xy[idx[:tr]], y[idx[:tr]]
    xe, ye = xy[idx[tr:]], y[idx[tr:]]
    to_t   = lambda a: torch.tensor(a, dtype=torch.float32)
    return (to_t(xt), to_t(yt)[:,None]), (to_t(xe), to_t(ye)[:,None])

# ----------------------------- model --------------------------------
class FC(nn.Module):
    def __init__(self, w, L):
        super().__init__()
        self.depth = L
        self.hid = nn.ModuleList()
        prev = 2
        for _ in range(L):
            l = nn.Linear(prev, w)
            nn.init.normal_(l.weight, 0., 1/math.sqrt(prev)); nn.init.zeros_(l.bias)
            self.hid.append(l); prev = w
        self.out = nn.Linear(prev, 1)
        nn.init.normal_(self.out.weight, 0., 1/math.sqrt(prev)); nn.init.zeros_(self.out.bias)
    def forward(self,x,*,hid=False,grad=False):
        for l in self.hid: x=torch.tanh(l(x))
        if hid:  return x                # last-hidden activity
        if grad: return self.out(x)      # to take gradient wrt params
        return torch.tanh(self.out(x))

# ---------------------     NEW: depth × width scaling    ---------------------
def per_layer_lr(layer, base_lr, depth):
    """
    µP / NTK rule of thumb:
       lr ∝ 1 / fan_in   (width scaling)
       lr ∝ 1 / depth    (depth scaling – keeps functional step O(1))
    """
    fan_in = layer.weight.data.size(1)          # columns
    return base_lr / (fan_in * depth)

def make_optim(net, base_lr, depth):
    """
    Builds param groups so that each weight matrix (and matching bias)
    gets its own learning‑rate η = base_lr / (fan_in · depth).
    """
    param_groups = []
    for mod in net.modules():
        if isinstance(mod, nn.Linear):
            lr = per_layer_lr(mod, base_lr, depth)
            param_groups.append({"params": [mod.weight], "lr": lr})
            if mod.bias is not None:
                param_groups.append({"params": [mod.bias], "lr": lr})
    return torch.optim.SGD(param_groups, momentum=0.0)

# ------------------------ helpers -----------------------------------
def batch_accuracy(out, y):
    """out, y  shape [B,1];   y is ±1"""
    return (torch.sign(out.detach()) == y).float().mean().item()

def loader_accuracy(net, loader):
    net.eval()
    total_correct, total = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            total_correct += (torch.sign(net(x)) == y).float().sum().item()
            total += y.size(0)
    return total_correct / total

'''
def train(net, loader, lr=0.05, epochs=200):
    opt, mse = torch.optim.SGD(net.parameters(),lr=lr), nn.MSELoss()
    net.train()
    for _ in range(epochs):
        for x,y in loader:
            opt.zero_grad(); mse(net(x),y).backward(); opt.step()
'''

def train_to_acc(net, loader, target=0.95, max_epochs=2000, base_lr=0.05, depth=1):
    """
    SGD until training accuracy ≥ target (default 90 %) or max_epochs reached.
    Returns the epoch at which the target was hit.
    """
    #opt, mse = torch.optim.SGD(net.parameters(), lr=lr), nn.MSELoss()
    opt = make_optim(net, base_lr, depth)       # <── changed
    mse = nn.MSELoss()
    for epoch in range(1, max_epochs + 1):
        net.train()
        for x, y in loader:
            opt.zero_grad()
            mse(net(x), y).backward()
            opt.step()
        if epoch % 10 == 0:                     # check every 10 epochs
            acc = loader_accuracy(net, loader)
            if acc >= target:
                print(f"  reached {acc*100:.1f}% at epoch {epoch}")
                return epoch                    # early success
    return max_epochs                            # may exit without hitting target

#def gram(mat):                        # R = HᵀH   or  K = GᵀG
#    return mat.T @ mat

# ---------------------     NEW: robust Gram operator     ---------------------
def gram(X, *, centre=False, l2_normalise=False):
    """
    Returns K = Φ Φᵀ with:
       • optional centring  : x ← x − mean(x)
       • optional L2‑normal.: x ← x / ‖x‖₂
    """
    if centre:
        X = X - X.mean(dim=0, keepdim=True)
    if l2_normalise:
        X = X / (X.norm(dim=1, keepdim=True) + 1e-9)
    return X @ X.T

def frob(A):                          # Frobenius norm
    return torch.norm(A, p='fro')

def alignment(GT, G0):                # trace(GT G0)/||·||/||·||
    return torch.trace(GT @ G0) / (frob(GT)*frob(G0) + 1e-8)

def rep_align(net, loader_probe, loader_train, *, base_lr=0.05):
    with torch.no_grad():
        H0 = torch.cat([net(x, hid=True) for x, _ in loader_probe], 0)
    R0 = gram(H0)

    train_to_acc(net, loader_train, depth=net.depth, base_lr=base_lr)

    with torch.no_grad():
        HT = torch.cat([net(x, hid=True) for x, _ in loader_probe], 0)
    RT = gram(HT)
    return alignment(RT, R0).item()


def tk_align(net, loader_probe, loader_train, *, subset=512, base_lr=0.05):
    # --- pick a subset from the *probe* data ---
    xb, _ = next(iter(DataLoader(loader_probe.dataset,
                                 batch_size=subset, shuffle=True)))

    params = [p for p in net.parameters() if p.requires_grad]
    def flat_grad(x):
        net.zero_grad()
        y = net(x.unsqueeze(0), grad=True)
        g = torch.autograd.grad(y, params, retain_graph=True)
        return torch.cat([gi.view(-1) for gi in g])

    # initial NTK
    G0 = torch.stack([flat_grad(x) for x in xb])
    K0 = gram(G0)

    # train
    train_to_acc(net, loader_train, depth=net.depth, base_lr=base_lr)

    # trained NTK
    GT = torch.stack([flat_grad(x) for x in xb])
    KT = gram(GT)
    return alignment(KT, K0).item()

def layerwise_RA(net, loader_probe, loader_train, *, base_lr=0.05):
    # 1. ---- collect initial activations on *probe* set -----------
    layer_buffers = [[] for _ in net.hid]
    hooks = [m.register_forward_hook(
             lambda m, _, o, idx=i: layer_buffers[idx].append(o.detach()))
             for i, m in enumerate(net.hid)]

    with torch.no_grad():
        for xb, _ in loader_probe:
            net(xb)

    acts_init = [torch.cat(buf, 0) for buf in layer_buffers]

    # 2. ---- train on *train* set ---------------------------------
    train_to_acc(net, loader_train, depth=net.depth, base_lr=base_lr)

    # 3. ---- collected trained activations ------------------------
    for buf in layer_buffers: buf.clear()
    with torch.no_grad():
        for xb, _ in loader_probe:
            net(xb)
    acts_tr = [torch.cat(buf, 0) for buf in layer_buffers]

    [h.remove() for h in hooks]

    return [alignment(gram(Ht), gram(H0)) for H0, Ht in zip(acts_init, acts_tr)]


def layerwise_KA(net, loader_probe, loader_train,
                 *, subset=512, base_lr=0.05):
    # pick subset from PROBE data
    xb, _ = next(iter(DataLoader(loader_probe.dataset,
                                 batch_size=subset, shuffle=True)))

    params = [p for p in net.parameters() if p.requires_grad]
    slices = np.cumsum([0] + [p.numel() for p in params])

    def flat_grad(x):
        net.zero_grad()
        y = net(x.unsqueeze(0), grad=True)
        g = torch.autograd.grad(y, params, retain_graph=True)
        return torch.cat([gi.view(-1) for gi in g])

    G0 = torch.stack([flat_grad(x) for x in xb])     # before training

    train_to_acc(net, loader_train, depth=net.depth, base_lr=base_lr)

    GT = torch.stack([flat_grad(x) for x in xb])     # after training

    KA = []
    for i in range(len(slices)-1):
        s, e  = slices[i], slices[i+1]
        K0_i  = G0[:, s:e] @ G0[:, s:e].T
        KT_i  = GT[:, s:e] @ GT[:, s:e].T
        KA_i  = alignment(KT_i, K0_i)
        KA.append(KA_i.item())
    return KA

# ------------------------ experiment --------------------------------
'''
def run(width, depth, trials=9, batch=8192):
    (xt,yt),(xe,ye)=make_circle()
    train_loader = DataLoader(TensorDataset(xt,yt), batch, shuffle=True)
    test_loader  = DataLoader(TensorDataset(xe,ye), batch, shuffle=False)
    RA, KA = [], []
    for s in trange(trials, desc=f"N={width} L={depth}"):
        torch.manual_seed(s); np.random.seed(s); random.seed(s)
        net = FC(width, depth)
        RA_i = layerwise_RA(net, test_loader)   # list[L]
        save_ckpt(net, width, depth, s)
        RA.append(np.mean(RA_i))                # ← average over layers
        torch.cuda.empty_cache()
        torch.manual_seed(s); np.random.seed(s); random.seed(s)
        ##################################################################
        #net = FC(width, depth)
        #KA_i = layerwise_KA(net, train_loader)  # list[L]
        ##################################################################
        net = FC(width, depth)
        net.load_state_dict(torch.load(
            f"{CHK_DIR}/net_w{width}_L{depth}_seed{s}.pt"))
        net.eval()
        KA_i = layerwise_KA(net, train_loader)  # re-uses saved weights
        ##################################################################
        KA.append(np.mean(KA_i))
        torch.cuda.empty_cache()
    return np.mean(RA), np.mean(KA)
'''
def run(width, depth, trials=9, batch=8192,
        base_lr=0.05, layerwise=False):
    (xt, yt), (xe, ye) = make_circle()
    loader_train = DataLoader(TensorDataset(xt, yt), batch, shuffle=True)
    loader_probe = DataLoader(TensorDataset(xe, ye), batch, shuffle=False)

    RA_all, KA_all = [], []          # scalar headline numbers
    RA_layers, KA_layers = [], []    # per‑layer lists (optional)

    for s in trange(trials, desc=f"W={width}  L={depth}"):
        torch.manual_seed(s); np.random.seed(s); random.seed(s)

        # ---------- build and save fresh model ----------
        net = FC(width, depth)
        save_ckpt(net, width, depth, s)

        # ---------- representation alignment ----------
        if layerwise:
            RA_i = layerwise_RA(net, loader_probe, loader_train)
            RA_layers.append(RA_i)               # list of length depth
            RA_all.append(np.mean(RA_i))
        else:
            RA_all.append(rep_align(net, loader_probe, loader_train))

        # ---------- tangent‑kernel alignment ----------
        net.load_state_dict(torch.load(
            f"{CHK_DIR}/net_w{width}_L{depth}_seed{s}.pt"))
        if layerwise:
            KA_i = layerwise_KA(net, loader_probe, loader_train)
            KA_layers.append(KA_i)
            KA_all.append(np.mean(KA_i))
        else:
            KA_all.append(tk_align(net, loader_probe, loader_train))

    headline_RA = np.mean(RA_all)
    headline_KA = np.mean(KA_all)

    # if you asked for layer‑wise results, also return their mean profile
    if layerwise:
        mean_RA_profile = np.mean(RA_layers, axis=0)  # shape [depth]
        mean_KA_profile = np.mean(KA_layers, axis=0)
        return headline_RA, headline_KA, mean_RA_profile, mean_KA_profile
    else:
        return headline_RA, headline_KA



if __name__ == "__main__":
    torch.set_grad_enabled(True)           # NTK needs grads
    #ra_rich, ka_rich = run(10, 12)
    #ra_lazy, ka_lazy = run(250, 2)
    #print(f"width=10,  depth=12,  RA = {ra_rich:.3f}  KA = {ka_rich:.3f}  (avg of 9)")
    #print(f"width=250, depth=2,   RA = {ra_lazy:.3f}  KA = {ka_lazy:.3f}  (avg of 9)")
    ra_narrow, ka_narrow, ra_prof_narrow, ka_prof_narrow = run(width=10, depth=4, layerwise=True)
    print("layer‑wise RA:", ra_prof_narrow) # [Layer 1 RA, Layer 2 RA ...]
    print("layer‑wise KA:", ka_prof_narrow) # [Layer 1 KA, Layer 1 Bias, Layer 2 KA, Layer 2 Bias ..., Output layer KA, Output layer Bias]
    #ra_wide, ka_wide, ra_prof_wide, ka_prof_wide = run(width=250, depth=2, layerwise=True)
    #print("layer‑wise RA:", ra_prof_wide)
    #print("layer‑wise KA:", ka_prof_wide)


# W = 10, L = 2
# layer‑wise RA: [0.9710357 0.8229258]
# layer‑wise KA: [0.84606728 0.64370983 0.61040685]

# W = 50, L = 2
# layer‑wise RA: [0.9982977  0.97693294]
# layer‑wise KA: [0.96850148 0.82529076 0.76006538]

# W = 250, L = 2
# layer‑wise RA: [0.99998355 0.99690807]
# layer‑wise KA: [0.99767243 0.96828882 0.95425314]

# W = 10, L = 4
# layer‑wise RA: [0.9910052  0.9212354  0.82818174 0.6983849 ]
# layer‑wise KA: [0.79621189 0.77488117 0.78056735 0.87252767 0.69415596 0.94384767
# 0.65240953 0.98345986 0.54127032 1.        ]