import math, random, argparse, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange
import os

# -------------- checkpoint helper -----------------------------------
CHK_DIR = "models"                 # top-level folder for weights
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

def train_to_acc(net, loader, target=0.95, max_epochs=2000, lr=0.05):
    """
    SGD until training accuracy ≥ target (default 90 %) or max_epochs reached.
    Returns the epoch at which the target was hit.
    """
    opt, mse = torch.optim.SGD(net.parameters(), lr=lr), nn.MSELoss()
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

def gram(mat):                        # R = HᵀH   or  K = GᵀG
    return mat.T @ mat

def frob(A):                          # Frobenius norm
    return torch.norm(A, p='fro')

def alignment(GT, G0):                # trace(GT G0)/||·||/||·||
    return torch.trace(GT @ G0) / (frob(GT)*frob(G0) + 1e-8)

def rep_align(net, loader):           # RA
    with torch.no_grad():
        H = torch.cat([net(x,hid=True) for x,_ in loader],0)   # [N × W]
    R0 = gram(H_init := H.clone())     # initial Gram
    #train(net, loader)                 # train in-place
    train_to_acc(net, loader)
    with torch.no_grad():
        H = torch.cat([net(x,hid=True) for x,_ in loader],0)
    RT = gram(H)
    return alignment(RT, R0).item()

def tk_align(net, loader, subset=512):     # KA (use subset for speed)
    idx = torch.randperm(len(loader.dataset))[:subset]
    sub_loader = DataLoader(loader.dataset, batch_size=subset, sampler=idx,
                            num_workers=0, shuffle=False)
    x_batch, _ = next(iter(sub_loader))
    # -- compute gradients wrt *all* parameters
    params = [p for p in net.parameters() if p.requires_grad]
    P = sum(p.numel() for p in params)
    def flatten_grads(grads):
        return torch.cat([g.reshape(-1) for g in grads], 0)    # [P]
    # initial NTK
    grads0 = []
    net.zero_grad()
    for xi in x_batch:
        y = net(xi.unsqueeze(0), grad=True)
        gs = torch.autograd.grad(y, params, retain_graph=True)
        grads0.append(flatten_grads(gs))
    G0 = torch.stack(grads0)           # [subset × P]
    K0 = gram(G0)
    # train and compute new NTK
    #train(net, loader)
    train_to_acc(net, loader)
    gradsT = []
    net.zero_grad()
    for xi in x_batch:
        y = net(xi.unsqueeze(0), grad=True)
        gs = torch.autograd.grad(y, params, retain_graph=True)
        gradsT.append(flatten_grads(gs))
    GT = torch.stack(gradsT)
    KT = gram(GT)
    return alignment(KT, K0).item()

def layerwise_RA(net, data_loader):
    """returns list[RA_i]"""
    acts_init, acts_trained = [], []

    # 1. register hooks -------------------------------------------------
    layer_outs = [[] for _ in net.hid]
    hooks = [layer.register_forward_hook(
             lambda m, _, o, idx=i: layer_outs[idx].append(o.detach()))
             for i, layer in enumerate(net.hid)]

    # 2. collect *initial* activations ---------------------------------
    with torch.no_grad():
        for xb, _ in data_loader:
            _ = net(xb.to(next(net.parameters()).device))
    acts_init = [torch.cat(o, 0) for o in layer_outs]

    # 3. train ----------------------------------------------------------
    #train(net, data_loader)                         # your existing train()
    train_to_acc(net, data_loader)

    # 4. collect *trained* activations ---------------------------------
    for lst in layer_outs: lst.clear()              # reset
    with torch.no_grad():
        for xb, _ in data_loader:
            _ = net(xb.to(next(net.parameters()).device))
    acts_trained = [torch.cat(o, 0) for o in layer_outs]

    # 5. RA_i per layer -------------------------------------------------
    RA = []
    for H0, HT in zip(acts_init, acts_trained):
        R0, RT = gram(H0), gram(HT)
        RA.append((torch.trace(RT @ R0) /
                   (frob(RT)*frob(R0) + 1e-8)).item())

    [h.remove() for h in hooks]
    return RA                                          # list of length L

def layerwise_KA(net, data_loader, subset=512):
    params = [p for p in net.parameters() if p.requires_grad]
    layer_param_slices = np.cumsum([0]+[p.numel() for p in params])

    # choose a subset of inputs for NTK estimation
    xb, _ = next(iter(DataLoader(data_loader.dataset,
                                 batch_size=subset, shuffle=True)))
    xb = xb.to(next(net.parameters()).device)

    # function to flatten gradient wrt **all** params
    def grad_flat(x):
        net.zero_grad()
        y = net(x.unsqueeze(0))
        grads = torch.autograd.grad(y, params, retain_graph=True)
        return torch.cat([g.reshape(-1) for g in grads])

    G_init, G_tr   = [], []
    # collect gradients *before* training
    for x in xb: G_init.append(grad_flat(x))
    G_init = torch.stack(G_init)                     # [subset × P]

    # train network
    #train(net, data_loader)
    train_to_acc(net, data_loader)

    # collect gradients *after* training
    for x in xb: G_tr.append(grad_flat(x))
    G_tr = torch.stack(G_tr)

    # per-layer KA
    KA = []
    for i in range(len(layer_param_slices)-1):
        s, e = layer_param_slices[i], layer_param_slices[i+1]
        K0_i = G_init[:, s:e] @ G_init[:, s:e].T
        KT_i = G_tr[:, s:e]   @ G_tr[:, s:e].T
        KA_i = torch.trace(KT_i @ K0_i) / (frob(KT_i)*frob(K0_i) + 1e-8)
        KA.append(KA_i.item())
    return KA

# ------------------------ experiment --------------------------------
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

if __name__ == "__main__":
    torch.set_grad_enabled(True)           # NTK needs grads
    ra_rich, ka_rich = run(10, 12)
    ra_lazy, ka_lazy = run(250, 2)
    print(f"width=10,  depth=12,  RA = {ra_rich:.3f}  KA = {ka_rich:.3f}  (avg of 9)")
    print(f"width=250, depth=2,   RA = {ra_lazy:.3f}  KA = {ka_lazy:.3f}  (avg of 9)")
