#!/usr/bin/env python
# ─────────────────────────────────────────────────────────────
#  MNIST • 3-D bottleneck visualisation with rotating GIF
# ─────────────────────────────────────────────────────────────
import os, math, random, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D        # noqa: F401 (needed for 3-D)
from tqdm import tqdm


# reproducibility & device -------------------------------------------------
SEED = 2024
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] device = {device}")

# hyper-parameters ----------------------------------------------------------
WIDTH, DEPTH      = 20, 16
BATCH_SIZE        = 8192
EPOCHS_BASE       = 200
LR_BASE_1, LR_BASE_2 = 1e-3, 3e-4
EPOCHS_BTL        = 120          # joint training schedule
LR_BTL            = 3e-4
WEIGHT_DECAY_BTL  = 1e-4
POINT_SIZE        = 6
GIF_N_FRAMES      = 72           # 5° per frame → 360°
GIF_NAME          = "mnist_ftle_3d.gif"
DATA_ROOT         = os.environ.get("MNIST_ROOT", "mnist")

# data (manual cache, no download) -----------------------------------------
mnist_train = datasets.MNIST(
    root=DATA_ROOT, train=True, download=False,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]))
train_loader = DataLoader(mnist_train, batch_size=BATCH_SIZE, shuffle=True)
print(f"[INFO] {len(mnist_train)} training images loaded.")

# helper: accuracy ----------------------------------------------------------
def accuracy(net, loader):
    net.eval(); hits = tot = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            hits += (torch.argmax(net(xb),1) == yb).sum().item()
            tot  += yb.numel()
    return hits / tot

# model definitions ---------------------------------------------------------
class BaseNet(nn.Module):                        # 16 × 20 tanh
    def __init__(self):
        super().__init__()
        self.hidden = nn.ModuleList()
        prev = 28*28
        for _ in range(DEPTH):
            l = nn.Linear(prev, WIDTH)
            nn.init.normal_(l.weight, 0., 1/math.sqrt(prev))
            nn.init.zeros_(l.bias)
            self.hidden.append(l);  prev = WIDTH
        self.out = nn.Linear(prev, 10)
        nn.init.normal_(self.out.weight, 0., 1/math.sqrt(prev))
        nn.init.zeros_(self.out.bias)
    def forward(self,x,*,return_last=False):
        x = x.view(x.size(0),-1)
        for l in self.hidden: x = torch.tanh(l(x))
        if return_last: return x
        return self.out(x)

class Bottleneck3D(nn.Module):                   # 20 → 3 → 10
    def __init__(self, base: BaseNet):
        super().__init__()
        self.encoder = base.hidden              # frozen
        self.bottle  = nn.Linear(20,3)
        self.out     = nn.Linear(3,10)
    def forward(self,x,*,return_bottle=False):
        x = x.view(x.size(0),-1)
        for l in self.encoder:
            with torch.no_grad():
                x = torch.tanh(l(x))
        b = self.bottle(x)
        if return_bottle: return b
        return self.out(torch.tanh(b))

# ──────────────────  stage 1: train base net  ─────────────────────────────
base = BaseNet().to(device)
opt  = torch.optim.Adam(base.parameters(), lr=LR_BASE_1)
ce   = nn.CrossEntropyLoss()
for ep in range(1, EPOCHS_BASE+1):
    if ep == 100:                               # LR decay
        for pg in opt.param_groups: pg["lr"] = LR_BASE_2
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        loss = ce(base(xb), yb);  loss.backward();  opt.step()
    if ep % 20 == 0 or ep==1:
        print(f"[base] ep {ep}/{EPOCHS_BASE}  acc={accuracy(base,train_loader)*100:.2f}%")
print(f"[base] final train acc ≈ {accuracy(base,train_loader)*100:.2f}%")

# ──────────────────  stage 2: train 3-D bottleneck  ───────────────────────
for p in base.hidden.parameters(): p.requires_grad_(False)
model = Bottleneck3D(base).to(device)
opt_btl = torch.optim.Adam(model.parameters(), lr=LR_BTL,
                           weight_decay=WEIGHT_DECAY_BTL)
for ep in range(1, EPOCHS_BTL+1):
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        opt_btl.zero_grad()
        loss = ce(model(xb), yb);  loss.backward();  opt_btl.step()
    if ep % 10 == 0 or ep==1:
        print(f"[bottle] ep {ep}/{EPOCHS_BTL}  acc={accuracy(model,train_loader)*100:.2f}%")

# ──────────────────  obtain 3-D coords & labels  ─────────────────────────
coords, labels = [], []
model.eval()
with torch.no_grad():
    for xb, yb in DataLoader(mnist_train, batch_size=BATCH_SIZE):
        coords.append(model(xb.to(device), return_bottle=True).cpu())
        labels.append(yb)
coords = torch.cat(coords).numpy()              # (60000,3)
labels = torch.cat(labels).numpy()

# ──────────────────  λ₁(x)  (skip if you have them)  ─────────────────────
print("[λ] computing maximal FTLE (may take ~20 min GPU)…")
def λ1(img):
    img = img.to(device).requires_grad_(True)
    last = base(img.unsqueeze(0), return_last=True).squeeze(0)
    J = torch.autograd.functional.jacobian(
            lambda z: base(z.unsqueeze(0), return_last=True).squeeze(0),
            img.view(-1))
    return (1/DEPTH)*torch.log(torch.linalg.svdvals(J).max()).item()
λ_vals=[λ1(img) for img,_ in tqdm(mnist_train)]
λ_vals=np.array(λ_vals); vmin,vmax=np.percentile(λ_vals,[1,99])

# ──────────────────  static 3-D scatter  & GIF  ──────────────────────────
fig = plt.figure(figsize=(6,6))
ax  = fig.add_subplot(111,projection='3d')
sc  = ax.scatter(coords[:,0],coords[:,1],coords[:,2],
                 c=λ_vals, cmap='coolwarm',
                 s=POINT_SIZE, vmin=vmin, vmax=vmax, lw=0)
ax.set_axis_off(); fig.colorbar(sc, shrink=0.6, pad=0.01,
                                label=r'$\lambda^{(L)}_1(x)$')
plt.tight_layout();  plt.savefig("mnist_ftle_3d.png",dpi=300)
print("Saved static view → mnist_ftle_3d.png")

# ───────── use Pillow instead of imageio to build the GIF ─────────
from PIL import Image
import io

GIF_N_FRAMES = 72        # one full turn (5° per frame)
GIF_FPS      = 12
GIF_NAME     = "mnist_ftle_3d.gif"

frames = []
for i in range(GIF_N_FRAMES):
    # rotate view
    ax.view_init(elev=20, azim=i * 360 / GIF_N_FRAMES)

    # save current canvas to an in-memory PNG
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=200, bbox_inches="tight", pad_inches=0)
    buf.seek(0)
    frames.append(Image.open(buf).convert("P", palette=Image.ADAPTIVE))

# write animated GIF
frames[0].save(
    GIF_NAME,
    save_all=True,
    append_images=frames[1:],
    duration=int(1000 / GIF_FPS),   # milliseconds / frame
    loop=0,
)
print(f"✓ rotating GIF saved as {GIF_NAME}")
