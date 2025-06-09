# -*- coding: utf-8 -*-
"""
Generate Figure 4 from
    L. Storm et al. (2024) “Finite-Time Lyapunov Exponents of Deep Neural Networks”

2025-06-08 • fully fixed, warning-free, GPU-aware
"""
import math, torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, matplotlib, os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
matplotlib.use("Agg")
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# ───────────────────────── configuration ────────────────────────────
SEED, LR = 2024, 0.05
BATCH_SIZE, EPOCHS_BASE, EPOCHS_HEAD = 8192, 200, 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
print(f"[INFO] device = {DEVICE}")

# ──────────────────────────── dataset ───────────────────────────────
def load_mnist():
    tfm = transforms.Compose([transforms.ToTensor(),
                              transforms.Lambda(lambda t: t.view(-1))])
    tr = datasets.MNIST("./data", True,  download=True, transform=tfm)
    te = datasets.MNIST("./data", False, download=True, transform=tfm)
    x_tr, y_tr = tr.data.float().view(-1,784)/255., tr.targets
    x_te, y_te = te.data.float().view(-1,784)/255., te.targets
    μ, σ = x_tr.mean(0,keepdims=True), x_tr.std(0,keepdims=True)
    return ((x_tr-μ)/σ).to(DEVICE), y_tr.to(DEVICE), \
           ((x_te-μ)/σ).to(DEVICE), y_te.to(DEVICE)

# ─────────────────────────── network ────────────────────────────────
class MnistNet(nn.Module):
    def __init__(self, width=20, depth=16, bottleneck=False):
        super().__init__()
        self.hidden = nn.ModuleList()
        prev = 784
        for _ in range(depth):
            l = nn.Linear(prev, width)
            nn.init.normal_(l.weight, 0., 1/math.sqrt(prev)); nn.init.zeros_(l.bias)
            self.hidden.append(l); prev = width
        self.bneck = nn.Linear(prev,2) if bottleneck else None
        if self.bneck is not None:
            nn.init.normal_(self.bneck.weight,0.,1/math.sqrt(prev)); nn.init.zeros_(self.bneck.bias)
            prev = 2
        self.out = nn.Linear(prev, 10); nn.init.zeros_(self.out.bias)

    def forward(self, x, *, ret_bneck=False, ret_hidden=False):
        for l in self.hidden: x = torch.tanh(l(x))
        hidden_out = x
        if self.bneck is not None:
            x = torch.tanh(self.bneck(x))
            if ret_bneck: return x
        if ret_hidden: return hidden_out
        return torch.tanh(self.out(x))

# ───────────────────────── training loop ────────────────────────────
def train(net, loader, *, epochs, desc):
    net.to(DEVICE); opt = torch.optim.SGD(net.parameters(), lr=LR)
    for _ in tqdm(range(epochs), desc=desc, leave=False):
        for xb,yb in loader:
            xb = xb.to(DEVICE); yb = F.one_hot(yb,10).float().to(DEVICE)
            opt.zero_grad(); loss = F.mse_loss(net(xb), yb); loss.backward(); opt.step()

# ───────────────────────── FTLE utilities ───────────────────────────
def spectral_norm(J, iters=10):
    v = torch.randn(J.shape[1], device=J.device)
    for _ in range(iters):
        v = F.normalize(J.T @ (J @ v), dim=0)
    return torch.linalg.norm(J @ v)

def ftle_batch(net, xs, *, depth, batch=256):
    lam = []; net.eval()
    if net.bneck is not None:
        readout = lambda z: net(z.unsqueeze(0), ret_bneck=True).squeeze(0)
    else:
        readout = lambda z: net(z.unsqueeze(0), ret_hidden=True).squeeze(0)
    for k in tqdm(range(0, len(xs), batch), desc="FTLE", leave=False):
        for v in xs[k:k+batch]:
            v = v.clone().detach().requires_grad_(True)
            J = torch.autograd.functional.jacobian(readout, v)
            lam.append((1/depth)*torch.log(spectral_norm(J)).cpu())
    return torch.stack(lam)

# ─────────────────────── adversarial helper ─────────────────────────
def fgsm_path(net,start,label,*,steps=40,α=0.01):
    img = start.clone().detach().to(DEVICE)
    tgt = F.one_hot(torch.tensor(label,device=DEVICE),10).float()
    path=[]
    for _ in range(steps):
        img.requires_grad_(True)
        out = net(img.unsqueeze(0)).squeeze(); loss = F.mse_loss(out,tgt)
        g, = torch.autograd.grad(loss,img); img = (img + α*g.sign()).detach()
        with torch.no_grad():
            path.append(net(img.unsqueeze(0),ret_bneck=True).squeeze().cpu())
    return torch.stack(path)

# ───────────────────────── figure-4 routine ─────────────────────────
def figure4():
    x_tr,y_tr,x_te,y_te = load_mnist()
    loader = DataLoader(TensorDataset(x_tr.cpu(),y_tr.cpu()),BATCH_SIZE,shuffle=True)

    base = MnistNet(20,16).to(DEVICE); train(base,loader,epochs=EPOCHS_BASE,desc="base")
    head = MnistNet(20,16,bottleneck=True).to(DEVICE)
    with torch.no_grad():
        for a,b in zip(base.hidden,head.hidden): b.weight.copy_(a.weight); b.bias.copy_(a.bias)
    for p in head.hidden.parameters(): p.requires_grad_(False)
    train(head,loader,epochs=EPOCHS_HEAD,desc="bottleneck")

    λ_tr = ftle_batch(head,x_tr,depth=16).numpy()
    λ_te = ftle_batch(head,x_te,depth=16).numpy()
    with torch.no_grad():
        proj = head(x_tr,ret_bneck=True).cpu().numpy()
        probs= head(x_te).cpu()
    H = (-probs*probs.log()).sum(1).numpy()
    errs=((head(x_te).argmax(1)!=y_te).float()*100).cpu().numpy()
    adv = fgsm_path(head,x_tr[(y_tr==9).nonzero()[0]],4).numpy()

    # panel (a)
    plt.figure(figsize=(6,6))
    sc=plt.scatter(proj[:,0],proj[:,1],c=λ_tr,cmap="coolwarm",s=5,vmin=-.1,vmax=.25)
    plt.colorbar(sc); plt.plot(adv[:,0],adv[:,1],c="orange",lw=2)
    plt.title("Figure 4a — bottleneck projection"); plt.savefig("figure4a_projection.png",dpi=300)

    # panel (b)
    order=np.argsort(λ_te); fig,ax=plt.subplots(figsize=(6,3))
    ax.plot(λ_te[order],errs[order],'k-'); ax.set_xlabel(r"$\lambda_1$"); ax.set_ylabel("error (%)")
    ax2=ax.twinx(); ax2.plot(λ_te[order],H[order],'g-'); ax2.set_ylabel("entropy H")
    fig.tight_layout(); fig.savefig("figure4b_error_uncertainty.png",dpi=300)

    # panels (c,d)
    widths=np.array([10,20,40,80,160]); means,stds=[],[]
    for N in tqdm(widths,desc="width sweep"):
        net=MnistNet(N,16).to(DEVICE); train(net,loader,epochs=200,desc=f"N{N}")
        sample=ftle_batch(net,x_tr[:5000],depth=16).numpy()
        means.append(sample.mean()); stds.append(sample.std())
    fig,(a1,a2)=plt.subplots(1,2,figsize=(10,3))
    a1.plot(widths,means,'k-o'); a1.set_xscale("log"); a1.set_title("Fig 4c mean λ")
    a2.plot(widths,stds,'k-o'); a2.set_xscale("log"); a2.set_title("Fig 4d std λ")
    fig.tight_layout(); fig.savefig("figure4cd_mean_std.png",dpi=300)

# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    figure4()
    print("\n[✓] Figure 4 PNGs saved.\n")
