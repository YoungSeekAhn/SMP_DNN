# train_dataset_multi_horizon.py
# ------------------------------------------------------------
# 1) 데이터셋 로드 (pickle) -> DataLoader
# 2) 다중입력 LSTM 모델 정의 (price/flow/fund 분기)
# 3) 훈련/검증 루프 + 조기종료 + 모델 저장
# 4) 테스트 평가
# ------------------------------------------------------------
import math
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset_functions import MultiInputTSDataset

# 데이터 로드
Code = "005930"
dataset_path = Path("datasets") / f"{Code}_dataset.pkl"
payload = pd.read_pickle(dataset_path)

ds_tr = payload["train"]
ds_va = payload["val"]
ds_te = payload["test"]
meta  = payload["meta"]

print("meta:", meta)

# DataLoader
def auto_bs(n, base=128):
    if n >= base: return base
    if n >= 64: return 64
    if n >= 32: return 32
    return max(8, n)

BATCH_SIZE = auto_bs(len(ds_tr))
NUM_WORKERS = 0

train_loader = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=False,  drop_last=False, num_workers=NUM_WORKERS)
val_loader   = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=NUM_WORKERS)
test_loader  = DataLoader(ds_te, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=NUM_WORKERS)

# 모델 정의
class BranchLSTM(nn.Module):
    def __init__(self, in_dim, hidden=64, layers=1, dropout=0.0):
        super().__init__()
        self.in_dim = in_dim
        if in_dim > 0:
            self.lstm = nn.LSTM(
                input_size=in_dim, hidden_size=hidden,
                num_layers=layers, batch_first=True,
                dropout=(dropout if layers > 1 else 0.0)
            )
            self.out_dim = hidden
        else:
            self.lstm = None
            self.out_dim = 0

    def forward(self, x):
        if self.in_dim == 0:
            return x.new_zeros(x.size(0), 0)
        _, (h_n, _) = self.lstm(x)
        return h_n[-1]

class MultiInputLSTM(nn.Module):
    def __init__(self, p_dim, f_dim, d_dim, hidden=64, layers=1,
                 head_hidden=128, out_dim=1, dropout=0.1):
        super().__init__()
        self.enc_price = BranchLSTM(p_dim, hidden=hidden, layers=layers, dropout=dropout)
        self.enc_flow  = BranchLSTM(f_dim, hidden=hidden, layers=layers, dropout=dropout)
        self.enc_fund  = BranchLSTM(d_dim, hidden=hidden, layers=layers, dropout=dropout)

        concat_dim = self.enc_price.out_dim + self.enc_flow.out_dim + self.enc_fund.out_dim
        self.mlp = nn.Sequential(
            nn.Linear(concat_dim, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, out_dim),
        )

    def forward(self, x_price, x_flow, x_fund):
        hp = self.enc_price(x_price)
        hf = self.enc_flow(x_flow)
        hd = self.enc_fund(x_fund)
        h  = torch.cat([hp, hf, hd], dim=-1)
        return self.mlp(h)

# 입력 차원 & out_dim
sample = ds_tr[0]
P = sample["x_price"].shape[-1]
Q = sample["x_flow"].shape[-1]
R = sample["x_fund"].shape[-1]
out_dim = len(meta["window"]["horizons"])
print(f"Output dimension (horizons): {out_dim}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiInputLSTM(p_dim=P, f_dim=Q, d_dim=R,
                       hidden=64, layers=1,
                       head_hidden=128, out_dim=out_dim,
                       dropout=0.1).to(device)

# 손실/최적화
target_kind = meta.get("target_kind", "logr")
criterion = nn.BCEWithLogitsLoss() if target_kind == "direction" else nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)

# 평가 함수
def evaluate(loader):
    model.eval()
    total_loss, n = 0.0, 0
    all_y, all_p = [], []
    with torch.no_grad():
        for batch in loader:
            xp = batch["x_price"].to(device)
            xf = batch["x_flow"].to(device)
            xd = batch["x_fund"].to(device)
            y  = batch["y"].to(device)

            pred = model(xp, xf, xd)
            loss = criterion(pred, y)
            total_loss += loss.item() * y.size(0)
            n += y.size(0)

            all_y.append(y.cpu())
            all_p.append(pred.cpu())

    import numpy as np
    y_all = torch.cat(all_y).numpy()
    p_all = torch.cat(all_p).numpy()

    metrics = {}
    if target_kind == "direction":
        y_hat = (p_all > 0).astype(int)
        y_bin = (y_all > 0.5).astype(int)
        metrics["acc"] = (y_hat == y_bin).mean()
    else:
        rmse = math.sqrt(((p_all - y_all) ** 2).mean())
        mae  = np.abs(p_all - y_all).mean()
        metrics["rmse"] = rmse
        metrics["mae"] = mae

    return total_loss / max(1, n), metrics

# 모델 훈련
from tqdm import tqdm


# 학습 함수
def train(num_epochs=30, grad_clip=1.0, es_patience=6):
    best_score = float("inf") if target_kind != "direction" else -float("inf")
    best_path = Path("models"); best_path.mkdir(exist_ok=True, parents=True)
    best_path = best_path / f"{Code}_best.pt"
    bad = 0
# ---- 기록용 history ----
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_rmse": [],
        "val_mae": [],
        "val_acc": []  # 분류일 때만 유효
    }
    
    for epoch in range(1, num_epochs+1):
        model.train()
        total, n = 0.0, 0

        print(f"\nEpoch {epoch}/{num_epochs}")
        pbar = tqdm(train_loader, desc="Training", leave=False)

        for batch in pbar:
            xp = batch["x_price"].to(device)
            xf = batch["x_flow"].to(device)
            xd = batch["x_fund"].to(device)
            y  = batch["y"].to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(xp, xf, xd)
            loss = criterion(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            total += loss.item() * y.size(0)
            n += y.size(0)

            # 진행 표시바에 현재 배치 loss 표시
            pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})

        train_loss = total / max(1, n)
        val_loss, val_metrics = evaluate(val_loader)
        scheduler.step(val_loss)

        # ---- history 기록 ----
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        if target_kind == "direction":
            history["val_acc"].append(val_metrics.get("acc", float("nan")))
        else:
            history["val_rmse"].append(val_metrics.get("rmse", float("nan")))
            history["val_mae"].append(val_metrics.get("mae", float("nan")))
                                      
        # Early Stopping 기준
        if target_kind == "direction":
            score = val_metrics["acc"]
            better = score > best_score
        else:
            score = val_metrics["rmse"]
            better = score < best_score

        if better:
            best_score = score
            bad = 0
            torch.save({"model": model.state_dict(),
                        "meta": meta,
                        "target_kind": target_kind}, best_path)
        else:
            bad += 1

        # Epoch 결과 출력
        if target_kind == "direction":
            print(f"[Epoch {epoch:02d}] Train {train_loss:.4f} | Val {val_loss:.4f} "
                  f"| Val Acc {val_metrics['acc']:.3f} | Best {best_score:.4f}")
        else:
            print(f"[Epoch {epoch:02d}] Train {train_loss:.4f} | Val {val_loss:.4f} "
                  f"| RMSE {val_metrics['rmse']:.4f} | MAE {val_metrics['mae']:.4f} "
                  f"| Best {best_score:.4f}")

        if bad >= es_patience:
            print("Early stopping.")
            break

    return best_path, history

# 실행
best_model_path, history = train(num_epochs=30)
model.load_state_dict(torch.load(best_model_path)["model"])
test_loss, test_metrics = evaluate(test_loader)

if target_kind == "direction":
    print(f"[TEST] loss {test_loss:.4f} | acc {test_metrics['acc']:.3f}")
else:
    print(f"[TEST] loss {test_loss:.4f} | rmse {test_metrics['rmse']:.4f} | mae {test_metrics['mae']:.4f}")
    

import matplotlib.pyplot as plt

epochs = range(1, len(history["train_loss"]) + 1)

# 1. Train vs Validation Loss
plt.figure(1,figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(epochs, history["train_loss"], label="Train Loss")
plt.plot(epochs, history["val_loss"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train vs Validation Loss")
plt.legend()
plt.grid(True)

# 2. Validation RMSE & MAE
plt.subplot(1,2,2)
plt.plot(epochs, history["val_rmse"], label="Val RMSE")
plt.plot(epochs, history["val_mae"], label="Val MAE")
plt.xlabel("Epoch")
plt.ylabel("Error")
plt.title("Validation RMSE & MAE")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()