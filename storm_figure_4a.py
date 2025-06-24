#[⚠️ Suspicious Content] #[⚠️ Suspicious Content] #!/usr/bin/env python
# ╭──────────────────────────────────────────────────────────────╮
# │  FIGURE 4 (a)  •  Maximal-FTLE field for MNIST (Storm 2024) │
# ╰──────────────────────────────────────────────────────────────╯
#  Requirements: torch ≥ 2.1, torchvision, numpy, matplotlib, tqdm

from __future__ import annotations
import os, math, random, numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

# ─────────────────────────  reproducibility  ─────────────────────────
SEED = 2024
torch.manual_seed(SEED);  np.random.seed(SEED);  random.seed(SEED)

# ───────────────────────────  device  ───────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO]  using device → {device}")

# ───────────────────  hyper-parameters (paper)  ─────────────────────
WIDTH       = 20          # neurons / hidden layer
DEPTH       = 16          # hidden layers
BATCH_SIZE  = 8_192
EPOCHS_BASE = 200         # Adam 1 e-3 → ≥ 98 % acc by 90-100 ep
LR_BASE_1   = 1e-3         # Adam for the first 100 ep
LR_BASE_2   = 3e-4         # smaller LR for fine-tuning
EPOCHS_BTL  = 30          # retrain bottleneck + soft-max
LR_BTL      = 3e-4
V_MIN, V_MAX = -0.10, 0.25    # colour-bar limits (paper)
WEIGHT_DECAY_BTL = 1e-4

# # ────────────────────────  data  (60 000 train)  ────────────────────
# mnist_train = datasets.MNIST(
#     "mnist", train=True, download=True,
#     transform=transforms.Compose([
#         transforms.ToTensor(),                        # [0,1]
#         transforms.Normalize((0.1307,), (0.3081,)),   # μ,σ
#     ]))
# train_loader = DataLoader(mnist_train, batch_size=BATCH_SIZE, shuffle=True)

# ─────────────────────  data  (no download)  ──────────────────────

DATA_ROOT = os.environ.get("MNIST_ROOT",
                           "mnist")  # <-- set your path

if not os.path.isdir(DATA_ROOT):
    raise RuntimeError(
        f"MNIST_ROOT directory '{DATA_ROOT}' does not exist.\n"
        "Create it or set the env-var MNIST_ROOT to the correct location.")

mnist_train = datasets.MNIST(
    root=DATA_ROOT,
    train=True,
    download=False,          #  ← crucial: never touch the internet
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]))

train_loader = DataLoader(mnist_train, batch_size=BATCH_SIZE, shuffle=True)
# sanity-check: if files are missing torchvision raises RuntimeError
print(f"[INFO]  MNIST found – {len(mnist_train)} training images loaded.")

# ╭────────────────────────  model definitions  ──────────────────────╮
class BaseNet(nn.Module):
    """16 hidden tanh layers (width-20) + 10-logit output."""
    def __init__(self, width=WIDTH, depth=DEPTH):
        super().__init__()
        self.hidden = nn.ModuleList()
        prev = 28*28
        for _ in range(depth):
            l = nn.Linear(prev, width)
            nn.init.normal_(l.weight, 0.0, 1/math.sqrt(prev))
            nn.init.zeros_(l.bias)
            self.hidden.append(l)
            prev = width
        self.out = nn.Linear(prev, 10)
        nn.init.normal_(self.out.weight, 0.0, 1/math.sqrt(prev))
        nn.init.zeros_(self.out.bias)

    def forward(self, x, *, return_last=False):
        x = x.view(x.size(0), -1)         # flatten 28×28 → 784
        for l in self.hidden:
            x = torch.tanh(l(x))
        if return_last:
            return x                      # (B, 20)
        return self.out(x)                # logits

class BottleneckNet(nn.Module):
    """All hidden layers frozen; 2-D bottleneck + new soft-max."""
    def __init__(self, base: BaseNet):
        super().__init__()
        self.frozen = base.hidden         # list[Linear] length 16
        self.bottle = nn.Linear(20, 2)
        self.out    = nn.Linear(2, 10)

    def forward(self, x, *, return_bottle=False):
        x = x.view(x.size(0), -1)
        for l in self.frozen:             # frozen tanh layers
            with torch.no_grad():
                x = torch.tanh(l(x))
        b = self.bottle(x)                # (B, 2)
        if return_bottle:
            return b
        return self.out(torch.tanh(b))

# ╰────────────────────────────────────────────────────────────────────╯

# ────────────────────────  stage 1 • train base  ────────────────────
base_net = BaseNet().to(device)
opt = torch.optim.Adam(base_net.parameters(), lr=LR_BASE_1)
ce  = nn.CrossEntropyLoss()

for ep in range(1, EPOCHS_BASE+1):
    # one-shot LR decay at half-time
    if ep == 100:
        for pg in opt.param_groups:
            pg["lr"] = LR_BASE_2
    running = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        loss = ce(base_net(xb), yb)
        loss.backward();  opt.step()
        running += loss.item() * xb.size(0)
    if ep % 10 == 0 or ep == 1:
        print(f"[base] epoch {ep:03d}/{EPOCHS_BASE}  loss = {running/len(mnist_train):.4f}")

# quick train-set accuracy
def accuracy(net, loader):
    net.eval(); hits = total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            hits  += (torch.argmax(net(xb), 1) == yb).sum().item()
            total += yb.numel()
    return hits / total
print(f"[RESULT]  base-net training accuracy ≈ {100*accuracy(base_net,train_loader):.2f} %")

# # ────────────────────  stage 2 • two-phase bottleneck  ──────────────────
# for p in base_net.hidden.parameters():          # freeze all hidden layers
#     p.requires_grad_(False)

# bnet = BottleneckNet(base_net).to(device)
# ce   = nn.CrossEntropyLoss()

# # ---------- phase 1: train only soft-max (2 → 10) -----------------------
# for p in bnet.bottle.parameters():              # freeze 20 → 2
#     p.requires_grad_(False)

# opt_soft = torch.optim.Adam(bnet.out.parameters(), lr=1e-3)
# for ep in range(1, 6):                          # 5 quick epochs
#     tot = 0.0
#     for xb, yb in train_loader:
#         xb, yb = xb.to(device), yb.to(device)
#         opt_soft.zero_grad()
#         loss = ce(bnet(xb), yb)
#         loss.backward(); opt_soft.step()
#         tot += loss.item()*xb.size(0)
#     print(f"[softmax-only] epoch {ep}/5  loss = {tot/len(mnist_train):.4f}")

# # ---------- phase 2: unfreeze bottleneck and fine-tune ------------------
# for p in bnet.bottle.parameters():
#     p.requires_grad_(True)

# EPOCHS_BTL = 80
# LR_BTL_1   = 3e-4
# LR_BTL_2   = 1e-4       # decay at epoch 60
# opt_btl = torch.optim.Adam(
#     bnet.parameters(),              # <- no filter needed
#     lr=LR_BTL_1, weight_decay=1e-4)

# for ep in range(1, EPOCHS_BTL+1):
#     if ep == 60:                                # one-shot LR decay
#         for pg in opt_btl.param_groups:
#             pg["lr"] = LR_BTL_2
#     tot = 0.0
#     for xb, yb in train_loader:
#         xb, yb = xb.to(device), yb.to(device)
#         opt_btl.zero_grad()
#         loss = ce(bnet(xb), yb)
#         loss.backward(); opt_btl.step()
#         tot += loss.item()*xb.size(0)
#     if ep % 10 == 0 or ep == 1:
#         acc = accuracy(bnet, train_loader)*100
#         print(f"[bottle] ep {ep:02d}/{EPOCHS_BTL}  loss={tot/len(mnist_train):.4f}  acc={acc:.2f}%")

# print(f"[RESULT]  final bottleneck accuracy = {accuracy(bnet, train_loader)*100:.2f}%")

# ───────────────────  bottleneck: train both layers together ──────────────────
for p in base_net.hidden.parameters():       # freeze all 16 hidden layers
    p.requires_grad_(False)

bnet = BottleneckNet(base_net).to(device)

EPOCHS_BTL = 180
LR_BTL     = 3e-4
opt_btl = torch.optim.Adam(bnet.parameters(), lr=LR_BTL, weight_decay=1e-4)
ce = nn.CrossEntropyLoss()

for ep in range(1, EPOCHS_BTL + 1):
    tot = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        opt_btl.zero_grad()
        loss = ce(bnet(xb), yb)
        loss.backward();  opt_btl.step()
        tot += loss.item() * xb.size(0)
    if ep % 10 == 0 or ep == 1:
        acc = accuracy(bnet, train_loader) * 100
        print(f"[bottle] ep {ep:03d}/{EPOCHS_BTL}  loss={tot/len(mnist_train):.4f}  acc={acc:.2f}%")

print(f"[RESULT]  final bottleneck accuracy = {accuracy(bnet, train_loader)*100:.2f}%")

# ---- report bottleneck accuracy -------------------------------------
def accuracy_loader(net, loader):
    net.eval(); hits = total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            hits  += (torch.argmax(net(xb), 1) == yb).sum().item()
            total += yb.numel()
    return hits / total

print(f"[RESULT]  bottleneck training accuracy ≈ "
      f"{100*accuracy_loader(bnet, train_loader):.2f} %")

# ───────────────  get 2-D coords for all 60 000 inputs  ─────────────
coords, labels = [], []
bnet.eval()
with torch.no_grad():
    for xb, yb in DataLoader(mnist_train, batch_size=BATCH_SIZE):
        coords.append(bnet(xb.to(device), return_bottle=True).cpu())
        labels.append(yb)
coords = torch.cat(coords).numpy()          # (60000, 2)
labels = torch.cat(labels).numpy()

# ───────────────  compute λ₁^(L)(x)  for all inputs  ────────────────
print("[λ] computing maximal FTLE for 60 000 images …")
def λ1(img: torch.Tensor) -> float:
    img = img.to(device).requires_grad_(True)
    last = base_net(img.unsqueeze(0), return_last=True).squeeze(0)  # (20,)
    J = torch.autograd.functional.jacobian(
            lambda z: base_net(z.unsqueeze(0), return_last=True).squeeze(0),
            img.view(-1))                                           # (20, 784)
    s_max = torch.linalg.svdvals(J).max()
    return (1/DEPTH) * torch.log(s_max).item()

λ_vals = []
for img, _ in tqdm(mnist_train):
    λ_vals.append(λ1(img))
λ_vals = np.array(λ_vals)

# ──────────────────  93 % bounding box for digit 0  ─────────────────
mask0 = (labels == 0)
xs0, ys0 = coords[mask0,0], coords[mask0,1]
p_low, p_high = 3.5, 96.5                  # leaves ≈93 %
x_l, x_r = np.percentile(xs0, [p_low, p_high])
y_b, y_t = np.percentile(ys0, [p_low, p_high])

# 3× zoom rectangle
cx, cy = (x_l+x_r)/2, (y_b+y_t)/2
w, h   = (x_r-x_l)*1.5, (y_t-y_b)*1.5     # half-width/height ×3

# ───────────────────────────  PLOT  ─────────────────────────────────
plt.figure(figsize=(8,6))
sc = plt.scatter(coords[:,0], coords[:,1], c=λ_vals,
                 cmap='coolwarm', s=4, linewidths=0,
                 vmin=V_MIN, vmax=V_MAX)
plt.colorbar(sc, label=r'$\lambda^{(L)}_{1}(x)$')

# main bounding box
plt.plot([x_l,x_r,x_r,x_l,x_l], [y_b,y_b,y_t,y_t,y_b], 'k-', lw=1)
# dashed 3× zoom rectangle
plt.plot([cx-w,cx+w,cx+w,cx-w,cx-w],
         [cy-h,cy-h,cy+h,cy+h,cy-h], 'k--', lw=1)

# ───────── label each digit cluster ─────────
import matplotlib.patheffects
for d in range(10):
    m = (labels == d)
    cx_d = np.median(coords[m,0]);  cy_d = np.median(coords[m,1])
    plt.text(cx_d, cy_d, str(d), ha='center', va='center',
             fontsize=14, fontweight='bold', color='black',
             path_effects=[matplotlib.patheffects.Stroke(
                              linewidth=3, foreground='white'),
                           matplotlib.patheffects.Normal()])

plt.title("Fig 4 (a) MNIST  colour = maximal FTLE", pad=10)
plt.xticks([]); plt.yticks([])
plt.tight_layout()
plt.savefig("figure4a_mnist_ftle.png", dpi=300)
