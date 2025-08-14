# train_dataset.py
# ------------------------------------------------------------
# 1) 데이터셋 로드 (pickle) -> DataLoader
# 2) 다중입력 LSTM 모델 정의 (price/flow/fund 분기)
# 3) 훈련/검증 루프 + 조기종료 + 모델 저장
# 4) 테스트 평가
# ------------------------------------------------------------
import os
from pathlib import Path
import math
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset_functions import MultiInputTSDataset  # noqa: F401

# ---- 1) 데이터셋 로드 ----
Code = "005930"  # 파일명에 사용한 코드
dataset_path = Path("datasets") / f"{Code}_dataset.pkl"
payload = pd.read_pickle(dataset_path)   # {'train': ds_tr, 'val': ds_va, 'test': ds_te, 'meta': meta}

ds_tr = payload["train"]
ds_va = payload["val"]
ds_te = payload["test"]
meta  = payload["meta"]

print("meta:", meta)

# 자동 배치 크기 (데이터가 적으면 자동 축소)
def auto_bs(n, base=128):
    if n >= base: return base
    if n >= 64: return 64
    if n >= 32: return 32
    return max(8, n)

BATCH_SIZE = auto_bs(len(ds_tr))
NUM_WORKERS = 0  # 파이썬 피클 객체이므로 0 권장(멀티프로세스 피클 이슈 피함)

train_loader = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True,  drop_last=True, num_workers=NUM_WORKERS)
val_loader   = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=NUM_WORKERS)
test_loader  = DataLoader(ds_te, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=NUM_WORKERS)

# ---- 2) 모델 정의 (분기별 LSTM -> concat -> MLP head) ----
class BranchLSTM(nn.Module):
    def __init__(self, in_dim, hidden=64, layers=1, dropout=0.0):
        super().__init__()
        self.in_dim = in_dim
        if in_dim > 0:
            self.lstm = nn.LSTM(input_size=in_dim, hidden_size=hidden,
                                num_layers=layers, batch_first=True, dropout=(dropout if layers>1 else 0.0))
            self.out_dim = hidden
        else:
            # 입력 특성이 0개일 때는 더미(0 벡터)로 대체
            self.lstm = None
            self.out_dim = 0

    def forward(self, x):  # x: [B, L, in_dim]
        if self.in_dim == 0:
            # [B, hidden] 대신 [B, 0] 반환
            return x.new_zeros(x.size(0), 0)
        _, (h_n, _) = self.lstm(x)   # h_n: [num_layers, B, hidden]
        return h_n[-1]               # [B, hidden] (마지막 레이어의 마지막 시점 hidden)

class MultiInputLSTM(nn.Module):
    def __init__(self, p_dim, f_dim, d_dim, hidden=64, layers=1, head_hidden=128, out_dim=1, dropout=0.1):
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

    def forward(self, x_price, x_flow, x_fund):  # [B,L,P],[B,L,Q],[B,L,R]
        hp = self.enc_price(x_price)
        hf = self.enc_flow(x_flow)
        hd = self.enc_fund(x_fund)
        h  = torch.cat([hp, hf, hd], dim=-1)  # [B, concat]
        y  = self.mlp(h)                      # [B, 1]
        return y

# 입력 차원은 샘플 하나로 파악
sample = ds_tr[0]
P = sample["x_price"].shape[-1]
Q = sample["x_flow"].shape[-1]
R = sample["x_fund"].shape[-1]
print(f"branch dims: price={P}, flow={Q}, fund={R}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiInputLSTM(p_dim=P, f_dim=Q, d_dim=R, hidden=64, layers=1, head_hidden=128, out_dim=1, dropout=0.1).to(device)

# ---- 3) 손실/최적화/스케줄러 ----
target_kind = meta.get("target_kind", "logr")
if target_kind == "direction":
    criterion = nn.BCEWithLogitsLoss()
else:
    criterion = nn.MSELoss()

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5, verbose=True)

# ---- 4) 학습/검증 루프 ----
def evaluate(loader):
    model.eval()
    total_loss, n = 0.0, 0
    all_y, all_p = [], []
    with torch.no_grad():
        for batch in loader:
            xp = batch["x_price"].to(device).float()
            xf = batch["x_flow"].to(device).float()
            xd = batch["x_fund"].to(device).float()
            y  = batch["y"].to(device).float()  # [B,1]

            pred = model(xp, xf, xd)  # [B,1]
            loss = criterion(pred, y)

            total_loss += loss.item() * y.size(0)
            n += y.size(0)
            all_y.append(y.detach().cpu())
            all_p.append(pred.detach().cpu())
    import torch as _t
    y_all = _t.cat(all_y, dim=0).squeeze(1).numpy()
    p_all = _t.cat(all_p, dim=0).squeeze(1).numpy()

    metrics = {}
    if target_kind == "direction":
        # 분류 지표
        import numpy as np
        y_hat = (p_all > 0).astype(int)  # 로짓>0 -> 1
        y_bin = (y_all > 0.5).astype(int)
        acc = (y_hat == y_bin).mean()
        metrics["acc"] = float(acc)
    else:
        # 회귀 지표
        import numpy as np
        rmse = math.sqrt(((p_all - y_all) ** 2).mean())
        mae  = np.abs(p_all - y_all).mean()
        metrics["rmse"] = float(rmse)
        metrics["mae"]  = float(mae)

    avg_loss = total_loss / max(1, n)
    return avg_loss, metrics

def train(num_epochs=30, grad_clip=1.0, es_patience=6):
    best_val = float("inf")
    best_path = Path("models"); best_path.mkdir(exist_ok=True, parents=True)
    best_path = best_path / f"{Code}_best.pt"
    bad = 0
    for epoch in range(1, num_epochs+1):
        model.train()
        total, n = 0.0, 0
        for batch in train_loader:
            xp = batch["x_price"].to(device).float()
            xf = batch["x_flow"].to(device).float()
            xd = batch["x_fund"].to(device).float()
            y  = batch["y"].to(device).float()

            optimizer.zero_grad(set_to_none=True)
            pred = model(xp, xf, xd)
            loss = criterion(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            total += loss.item() * y.size(0)
            n += y.size(0)

        train_loss = total / max(1, n)
        val_loss, val_metrics = evaluate(val_loader)
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            bad = 0
            torch.save({"model": model.state_dict(),
                        "meta": meta,
                        "target_kind": target_kind}, best_path)
        else:
            bad += 1

        # 로그
        if target_kind == "direction":
            print(f"[{epoch:02d}] train {train_loss:.4f} | val {val_loss:.4f} | acc {val_metrics.get('acc',0):.3f} | best {best_val:.4f}")
        else:
            print(f"[{epoch:02d}] train {train_loss:.4f} | val {val_loss:.4f} | "
                  f"rmse {val_metrics.get('rmse',0):.4f} mae {val_metrics.get('mae',0):.4f} | best {best_val:.4f}")

        if bad >= es_patience:
            print("Early stopping.")
            break

    print("Best model:", best_path)
    return best_path

best_model_path = train(num_epochs=30)

# ---- 5) 테스트 평가 ----
ckpt = torch.load(best_model_path, map_location=device)
model.load_state_dict(ckpt["model"])
test_loss, test_metrics = evaluate(test_loader)

if target_kind == "direction":
    print(f"[TEST] loss {test_loss:.4f} | acc {test_metrics['acc']:.3f}")
else:
    print(f"[TEST] loss {test_loss:.4f} | rmse {test_metrics['rmse']:.4f} | mae {test_metrics['mae']:.4f}")