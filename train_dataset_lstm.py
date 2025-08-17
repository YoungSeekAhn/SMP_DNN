# train_dataset.py
import math
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt

# ---------------- Load payload ----------------
Code = "005930"  # 파일명 접두
payload_path = Path("datasets") / f"{Code}_dataset.pkl"
payload = pd.read_pickle(payload_path)

meta = payload["meta"]
feat = meta["feature_config"]
lookback = meta["window"]["lookback"]
horizons = meta["window"]["horizons"]
target_kind = meta.get("target_kind", "logr")

Xp_tr, Xf_tr, Xd_tr, Y_tr = payload["Xp_tr"], payload["Xf_tr"], payload["Xd_tr"], payload["Y_tr"]
Xp_va, Xf_va, Xd_va, Y_va = payload["Xp_va"], payload["Xf_va"], payload["Xd_va"], payload["Y_va"]
Xp_te, Xf_te, Xd_te, Y_te = payload["Xp_te"], payload["Xf_te"], payload["Xd_te"], payload["Y_te"]

# ---------------- Dataset ----------------
class MIDataset(Dataset):
    def __init__(self, Xp, Xf, Xd, Y):
        self.Xp, self.Xf, self.Xd, self.Y = Xp, Xf, Xd, Y
    def __len__(self): return len(self.Y)
    def __getitem__(self, idx):
        return {
            "x_price": torch.from_numpy(self.Xp[idx]),
            "x_flow":  torch.from_numpy(self.Xf[idx]),
            "x_fund":  torch.from_numpy(self.Xd[idx]),
            "y":       torch.from_numpy(self.Y[idx]),
        }

ds_tr = MIDataset(Xp_tr, Xf_tr, Xd_tr, Y_tr)
ds_va = MIDataset(Xp_va, Xf_va, Xd_va, Y_va)
ds_te = MIDataset(Xp_te, Xf_te, Xd_te, Y_te)

def auto_bs(n, base=128):
    if n >= base: return base
    if n >= 64: return 64
    if n >= 32: return 32
    return max(8, n)

train_loader = DataLoader(ds_tr, batch_size=auto_bs(len(ds_tr)), shuffle=True,  drop_last=False)
val_loader   = DataLoader(ds_va, batch_size=auto_bs(len(ds_va)), shuffle=False, drop_last=False)
test_loader  = DataLoader(ds_te, batch_size=auto_bs(len(ds_te)), shuffle=False, drop_last=False)

# ---------------- Model ----------------
class BranchLSTM(nn.Module):
    def __init__(self, in_dim, hidden=96, layers=1, dropout=0.2):
        super().__init__()
        self.in_dim = in_dim
        if in_dim > 0:
            self.lstm = nn.LSTM(in_dim, hidden, num_layers=layers, batch_first=True,
                                dropout=(dropout if layers > 1 else 0.0))
            self.out_dim = hidden
        else:
            self.lstm = None; self.out_dim = 0
    def forward(self, x):
        if self.in_dim == 0: return x.new_zeros(x.size(0), 0)
        _, (h, _) = self.lstm(x)
        return h[-1]


class MultiInputLSTM(nn.Module):
    def __init__(self, p_dim, f_dim, d_dim, hidden=96, layers=1, head_hidden=128, out_dim=1, dropout=0.2):
        super().__init__()
        self.enc_p = BranchLSTM(p_dim, hidden, layers, dropout)
        self.enc_f = BranchLSTM(f_dim, hidden, layers, dropout)
        self.enc_d = BranchLSTM(d_dim, hidden, layers, dropout)
        cat_dim = self.enc_p.out_dim + self.enc_f.out_dim + self.enc_d.out_dim
        self.mlp = nn.Sequential(
            nn.Linear(cat_dim, head_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(head_hidden, out_dim)
        )
    def forward(self, xp, xf, xd):
        h = torch.cat([self.enc_p(xp), self.enc_f(xf), self.enc_d(xd)], dim=-1)
        return self.mlp(h)

sample = ds_tr[0]
P = sample["x_price"].shape[-1]
F = sample["x_flow"].shape[-1]
D = sample["x_fund"].shape[-1]
H = sample["y"].shape[-1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MultiInputLSTM(P, F, D, hidden=96, layers=1, head_hidden=128, out_dim=H, dropout=0.2).to(device)
criterion = nn.MSELoss()  # (logr/pct/close 모두 MSE로 회귀)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.5e-3, weight_decay=0.5e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)

# ---------------- Eval ----------------
def evaluate(loader):
    model.eval(); total=0; n=0; Ys=[]; Ps=[]
    with torch.no_grad():
        for b in loader:
            xp=b["x_price"].to(device); xf=b["x_flow"].to(device); xd=b["x_fund"].to(device)
            y=b["y"].to(device)
            p=model(xp,xf,xd)
            loss=criterion(p,y)
            total+=loss.item()*y.size(0); n+=y.size(0)
            Ys.append(y.cpu()); Ps.append(p.cpu())
    Y=torch.cat(Ys,0).numpy(); P=torch.cat(Ps,0).numpy()
    rmse=math.sqrt(((P-Y)**2).mean()); mae=np.abs(P-Y).mean()
    return total/max(1,n), {"rmse":rmse,"mae":mae}

# ---------------- Train ----------------
def train(num_epochs=20, grad_clip=1.0, es_patience=6, plot=True, save_fig="curves.png"):
    best=float("inf"); bad=0
    Path("models").mkdir(exist_ok=True, parents=True)
    best_path=Path("models")/f"{meta['Code']}_best.pt"

    history={"train_loss":[],"val_loss":[],"val_rmse":[],"val_mae":[]}

    for epoch in range(1, num_epochs+1):
        model.train(); total=0; n=0
        print(f"\nEpoch {epoch}/{num_epochs}")
        pbar=tqdm(train_loader, desc="Training", leave=False)
        for b in pbar:
            xp=b["x_price"].to(device); xf=b["x_flow"].to(device); xd=b["x_fund"].to(device)
            y=b["y"].to(device)
            optimizer.zero_grad(set_to_none=True)
            p=model(xp,xf,xd); loss=criterion(p,y)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            total+=loss.item()*y.size(0); n+=y.size(0)
            pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})

        tr_loss=total/max(1,n)
        va_loss, va_m = evaluate(val_loader)
        scheduler.step(va_loss)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["val_rmse"].append(va_m["rmse"])
        history["val_mae"].append(va_m["mae"])

        print(f"[Epoch {epoch:02d}] Train {tr_loss:.4f} | Val {va_loss:.4f} | RMSE {va_m['rmse']:.4f} | MAE {va_m['mae']:.4f} | Best {best:.4f}")

        if va_m["rmse"]<best:
            best=va_m["rmse"]; bad=0
            torch.save({"model": model.state_dict(), "target_kind": target_kind}, best_path)
        else:
            bad+=1
        if bad>=es_patience:
            print("Early stopping."); break

    if plot:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1,2, figsize=(12,5))
        axes[0].plot(history["train_loss"], label="Train"); axes[0].plot(history["val_loss"], label="Val")
        axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(True)
        axes[1].plot(history["val_rmse"], label="Val RMSE"); axes[1].plot(history["val_mae"], label="Val MAE")
        axes[1].set_title("Val Errors"); axes[1].legend(); axes[1].grid(True)
        plt.tight_layout(); plt.savefig(save_fig, dpi=150); plt.show()
        print(f"Saved curves -> {save_fig}")

    return best_path, history

if __name__ == "__main__":
    best_model_path, history = train(num_epochs=30)
    ckpt = torch.load(best_model_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    te_loss, te_m = evaluate(test_loader)
    print(f"[TEST] loss {te_loss:.4f} | rmse {te_m['rmse']:.4f} | mae {te_m['mae']:.4f}")
