# evaluate_ops_report.py
# ------------------------------------------------------------
# 목적:
# - 모델 로드 → Test 예측 수집 (P, Y)
# - 베이스라인 대비 개선률, 방향정확도(Hit-rate)
# - 거래비용 포함 백테스트(간단 신호: 임계값/볼 타게팅) → 에쿼티/드로우다운/롤링 샤프
# - 시각화: Aligned 절대가격(예측 vs 실제), 에쿼티, 드로우다운, 롤링 샤프,
#           Decile 리프트(스코어 상위군 유효성), 예측-실현 산포도
# ------------------------------------------------------------
import os, pickle, math
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ========= 설정 =========
Code = "005930"
BATCH_SIZE = 256
COST_BPS = 15           # 왕복 거래비용 가정 (bps)
VOL_WIN = 20            # 롤링 변동성 창(거래 포지션 스케일링용)
LAMBDA_MOM = 0.2        # 포지션 관성(0~1), 0.2 권장
MAX_LEV = 1.0           # 최대 노출(롱/숏 클리핑 한도)
K_VOL = 2.0             # r_exp / (K * sigma) 스케일 상수
ROLLING_H1_PLOT = True  # h=1 롤링 경로도 함께 시각화할지
# =======================

# ========= 공용 유틸 =========
def load_payload(code: str):
    path = Path("datasets") / f"{code}_dataset.pkl"
    assert path.exists(), f"payload not found: {path}"
    return pd.read_pickle(path)  # {'Xp_tr', 'Xf_tr', 'Xd_tr', 'Y_tr', ..., 'meta'}

class MIDataset(Dataset):
    def __init__(self, Xp, Xf, Xd, Y):
        self.Xp = Xp.astype(np.float32)
        self.Xf = Xf.astype(np.float32)
        self.Xd = Xd.astype(np.float32)
        self.Y  = Y.astype(np.float32)
    def __len__(self): return len(self.Y)
    def __getitem__(self, i):
        return {
            "x_price": torch.from_numpy(self.Xp[i]),
            "x_flow":  torch.from_numpy(self.Xf[i]),
            "x_fund":  torch.from_numpy(self.Xd[i]),
            "y":       torch.from_numpy(self.Y[i]),
        }

class BranchLSTM(nn.Module):
    def __init__(self, in_dim, hidden=96, layers=1, dropout=0.2):
        super().__init__()
        self.in_dim = in_dim
        if in_dim > 0:
            self.lstm = nn.LSTM(in_dim, hidden, num_layers=layers, batch_first=True,
                                dropout=(dropout if layers>1 else 0.0))
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
        cat = self.enc_p.out_dim + self.enc_f.out_dim + self.enc_d.out_dim
        self.mlp = nn.Sequential(
            nn.Linear(cat, head_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(head_hidden, out_dim)
        )
    def forward(self, xp, xf, xd):
        h = torch.cat([self.enc_p(xp), self.enc_f(xf), self.enc_d(xd)], dim=-1)
        return self.mlp(h)

def build_model_from_shapes(Xp, Xf, Xd, Y, device):
    P = Xp.shape[-1]; F = Xf.shape[-1]; D = Xd.shape[-1]; H = Y.shape[-1]
    m = MultiInputLSTM(P, F, D, hidden=96, layers=1, head_hidden=128, out_dim=H, dropout=0.2).to(device)
    return m

def get_close_idx(feat: Dict[str, List[str]]) -> int:
    cols = feat.get("price_cols", [])
    return cols.index("close") if "close" in cols else 3

def to_growth(a: np.ndarray, kind: str) -> np.ndarray:
    if kind == "logr": return np.exp(a)
    if kind == "pct":  return 1.0 + a
    if kind == "close": return None
    raise ValueError("target_kind must be in {'logr','pct','close'}")

def to_simple_return(a: np.ndarray, kind: str) -> np.ndarray:
    """모델 출력/정답(라벨)을 '단순수익률'로 변환 (h=1 열을 주로 사용)"""
    if kind == "logr": return np.exp(a) - 1.0
    if kind == "pct":  return a
    raise ValueError("target_kind='close'는 (pred_price / P_t - 1)로 변환 필요")

def rolling_sharpe(r: np.ndarray, win=63) -> np.ndarray:
    s = pd.Series(r)
    mu = s.rolling(win).mean() * np.sqrt(252)
    sd = s.rolling(win).std()  * np.sqrt(252)
    return (mu / (sd + 1e-12)).values

def drawdown_curve(equity: np.ndarray) -> np.ndarray:
    cummax = np.maximum.accumulate(equity)
    return equity / (cummax + 1e-12) - 1.0

# ===== horizon 정렬 시계열 (예측/실제 모두 생성) =====
def build_aligned_series_from_base(preds, trues, base_closes, horizons, target_kind):
    if not isinstance(horizons, (list, tuple)): horizons=[int(horizons)]
    Hs = list(horizons); h_max = max(Hs)
    N, H = preds.shape; assert trues.shape == preds.shape
    aligned_pred = {h: np.full(N+h_max, np.nan, float) for h in Hs}
    aligned_true = {h: np.full(N+h_max, np.nan, float) for h in Hs}

    if target_kind in ("logr","pct"):
        g_pred = to_growth(preds, target_kind)  # (N,H)
        g_true = to_growth(trues, target_kind)  # (N,H)
        for j, h in enumerate(Hs):
            for k in range(N):
                t = k + h
                if t < N + h_max:
                    aligned_pred[h][t] = base_closes[k] * g_pred[k, j]
                    aligned_true[h][t] = base_closes[k] * g_true[k, j]
    else:
        # 'close' 라벨을 스케일 없이 썼다는 전제. 스케일했다면 여기서 inverse_transform 필요.
        for j, h in enumerate(Hs):
            for k in range(N):
                t = k + h
                if t < N + h_max:
                    aligned_pred[h][t] = preds[k, j]
                    aligned_true[h][t] = trues[k, j]
    return aligned_pred, aligned_true

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) 데이터/메타
    payload = load_payload(Code)
    meta = payload["meta"]
    feat = meta["feature_config"]
    horizons = meta["window"]["horizons"]
    target_kind = meta.get("target_kind", "logr")
    close_idx = get_close_idx(feat)

    # 2) Test loader
    ds_te = MIDataset(payload["Xp_te"], payload["Xf_te"], payload["Xd_te"], payload["Y_te"])
    test_loader = DataLoader(ds_te, batch_size=BATCH_SIZE, shuffle=False)

    # 3) 모델 로드
    model = build_model_from_shapes(payload["Xp_tr"], payload["Xf_tr"], payload["Xd_tr"], payload["Y_tr"], device)
    ckpt_path = Path("models") / f"{Code}_best.pt"
    assert ckpt_path.exists(), f"checkpoint not found: {ckpt_path}"
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"]); model.eval()
    target_kind = ckpt.get("target_kind", target_kind)

    # 4) 스케일러 로드
    scalers_path = Path("artifacts") / "scalers.pkl"
    assert scalers_path.exists(), f"scalers not found: {scalers_path}"
    with open(scalers_path, "rb") as f:
        scalers = pickle.load(f)
    price_scaler = scalers.get("price", None)

    # 5) 예측/정답/베이스클로즈 수집
    Ps, Ys, base_closes_list = [], [], []
    with torch.no_grad():
        for b in test_loader:
            p = model(b["x_price"].to(device), b["x_flow"].to(device), b["x_fund"].to(device))
            Ps.append(p.cpu().numpy())
            Ys.append(b["y"].cpu().numpy())

            x_last = b["x_price"][:, -1, :].numpy()  # (B, P)
            if price_scaler is not None:
                inv = price_scaler.inverse_transform(x_last)
                base_closes_list.append(inv[:, close_idx])
            else:
                base_closes_list.append(x_last[:, close_idx])
    P = np.concatenate(Ps, 0)   # (N,H)
    Y = np.concatenate(Ys, 0)   # (N,H)
    base_closes = np.concatenate(base_closes_list, 0)  # (N,)
    N, H = P.shape

    # 6) 기본 성능 (RMSE/MAE 전체 + Horizon별)
    rmse_all = math.sqrt(((P - Y) ** 2).mean())
    mae_all  = np.abs(P - Y).mean()
    rmse_h = [math.sqrt(((P[:, j] - Y[:, j]) ** 2).mean()) for j in range(H)]
    mae_h  = [np.abs(P[:, j] - Y[:, j]).mean() for j in range(H)]
    print(f"[TEST] RMSE {rmse_all:.4f} | MAE {mae_all:.4f}")
    print("Test per-horizon RMSE:", np.round(rmse_h, 4).tolist())
    print("Test per-horizon MAE :", np.round(mae_h, 4).tolist())

    # 7) 베이스라인 대비 개선률
    if target_kind in ("logr","pct"):
        zero_pred = np.zeros_like(P)              # 항상 0 수익률
        rmse_base = math.sqrt(((zero_pred - Y) ** 2).mean())
        mae_base  = np.abs(zero_pred - Y).mean()
        impr_rmse = 100.0 * (rmse_base - rmse_all) / (rmse_base + 1e-12)
        impr_mae  = 100.0 * (mae_base  - mae_all ) / (mae_base  + 1e-12)
        print(f"Baseline (ZeroReturn) RMSE {rmse_base:.4f} | MAE {mae_base:.4f}")
        print(f"Improvement vs Baseline: RMSE {impr_rmse:.2f}% | MAE {impr_mae:.2f}%")
    else:
        print("[NOTE] target_kind='close'는 ZeroReturn 베이스라인이 부적절합니다. (직전가 유지 등으로 별도 구성 필요)")

    # 8) 방향 정확도(Hit-rate, 각 horizon)
    if target_kind in ("logr","pct"):
        hit = []
        for j in range(H):
            hp = np.sign(P[:, j]); ht = np.sign(Y[:, j])
            # 0인 경우는 제외할 수도 있으나 여기선 포함
            hit.append((hp == ht).mean())
        print("Hit-rate by horizon:", [round(float(x), 3) for x in hit])
    else:
        print("[NOTE] 'close' 타깃은 가격 수준을 예측하므로 hit-rate(방향)는 별도 정의가 필요합니다.")

    # 9) h=1 기준 운용 백테스트 (단순 임계값 + 볼 타게팅 혼합)
    #    기대수익 r_exp(단순수익률), 실제수익 r_real 계산
    if target_kind in ("logr","pct"):
        h_idx = horizons.index(1) if 1 in horizons else 0
        r_exp = to_simple_return(P[:, h_idx], target_kind)     # 기대(예측) 수익률
        r_real = to_simple_return(Y[:, h_idx], target_kind)    # 실현(정답) 수익률

        # (a) 임계값: 상하위 quantile로 롱/숏/중립 분기
        qL, qS = np.quantile(r_exp, [0.3, 0.7])  # 간단 예시 (튜닝 대상)
        sig_thr = np.where(r_exp >= qS, 1, np.where(r_exp <= qL, -1, 0)).astype(float)

        # (b) 볼 타게팅: pos_target = clip( r_exp / (K * sigma), -MAX_LEV, MAX_LEV )
        logr_real = np.log1p(r_real)
        sigma = pd.Series(logr_real).rolling(VOL_WIN).std().bfill().values
        pos_vol = np.clip(r_exp / (K_VOL * (sigma + 1e-8)), -MAX_LEV, MAX_LEV)

        # (c) 혼합 포지션: 50% 임계값, 50% 볼 타게팅 (예시)
        pos_target = 0.5 * sig_thr + 0.5 * pos_vol
        # 관성 적용(거래 줄이기)
        pos = np.zeros_like(pos_target)
        for t in range(1, len(pos)):
            pos[t] = (1 - LAMBDA_MOM) * pos[t-1] + LAMBDA_MOM * pos_target[t]

        # 체결/비용: 다음 스텝 실현수익 반영, turnover * COST_BPS 비용 차감
        turnover = np.abs(np.diff(pos, prepend=0.0))
        r_net = pos[:-1] * r_real[1:] - turnover[1:] * (COST_BPS / 10000.0)
        equity = (1 + r_net).cumprod()
        bench  = (1 + r_real[1:]).cumprod()

        # 지표 요약
        sharpe = (np.mean(r_net) / (np.std(r_net) + 1e-12)) * np.sqrt(252)
        max_dd = drawdown_curve(equity).min()
        cagr   = (equity[-1] ** (252.0 / max(1, len(r_net))) - 1.0) if len(r_net)>0 else 0.0
        print(f"[Backtest h=1] Sharpe {sharpe:.2f} | MaxDD {max_dd:.2%} | CAGR {cagr:.2%}")

        # ---- 플롯 1: 에쿼티 vs 벤치마크
        plt.figure(figsize=(12,5))
        plt.plot(equity, label="Strategy (net)")
        plt.plot(bench,  label="Buy&Hold")
        plt.title(f"Equity Curve (cost={COST_BPS}bps, λ={LAMBDA_MOM}, K={K_VOL})")
        plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()

        # ---- 플롯 2: 드로우다운
        dd = drawdown_curve(equity)
        plt.figure(figsize=(12,3.5))
        plt.fill_between(range(len(dd)), dd, 0, alpha=0.4)
        plt.title("Drawdown"); plt.tight_layout(); plt.show()

        # ---- 플롯 3: 롤링 샤프
        rs = rolling_sharpe(r_net, win=63)
        plt.figure(figsize=(12,3.5))
        plt.plot(rs); plt.axhline(0, linewidth=1)
        plt.title("Rolling Sharpe (3M)"); plt.tight_layout(); plt.show()
    else:
        print("[SKIP Backtest] target_kind='close'는 r_exp/r_real 정의가 다릅니다. (pred/P_t - 1)로 변환 후 동일 로직 적용하세요.)")

    # 10) Aligned 절대가격: 예측 vs 실제 (모든 horizon)
    aligned_pred, aligned_true = build_aligned_series_from_base(P, Y, base_closes, horizons, target_kind)
    plt.figure(figsize=(13,5))
    for h in horizons:
        plt.plot(aligned_true[h], linestyle="--", alpha=0.7, label=f"True h={h}")
        plt.plot(aligned_pred[h], alpha=0.9, label=f"Pred h={h}")
    plt.title(f"[Aligned] Absolute Price by Horizon (Code={Code}, target={target_kind})")
    plt.xlabel("Test timeline index"); plt.ylabel("Price"); plt.grid(True)
    plt.legend(ncol=2); plt.tight_layout(); plt.show()

    # 11) Decile 리프트(스코어 상위군 유효성): h=1 기준
    if target_kind in ("logr","pct"):
        # 예측 점수 = r_exp, 실현 = r_real(+1 시프트로 체결 반영)
        score = r_exp
        r_next = r_real  # decile은 동시점 평균 비교 용도로 그대로도 OK
        bins = pd.qcut(score, 10, labels=False, duplicates='drop')
        lift = pd.Series(r_next).groupby(bins).mean().values
        plt.figure(figsize=(8,4))
        plt.plot(lift, marker="o")
        plt.title("Decile Lift (mean realized return by score decile)")
        plt.xlabel("Decile (low→high)"); plt.ylabel("Mean realized return")
        plt.grid(True); plt.tight_layout(); plt.show()

        # 12) 예측-실현 산포도
        plt.figure(figsize=(6,6))
        plt.scatter(score, r_next, s=10, alpha=0.35)
        plt.title("Prediction vs Realized (h=1)")
        plt.xlabel("Predicted simple return"); plt.ylabel("Realized simple return")
        plt.axhline(0, linewidth=1, alpha=0.5); plt.axvline(0, linewidth=1, alpha=0.5)
        plt.tight_layout(); plt.show()

    # 13) (옵션) rolling h=1 경로
    if ROLLING_H1_PLOT and target_kind in ("logr","pct"):
        P0 = float(base_closes[0])
        h_idx = horizons.index(1) if 1 in horizons else 0
        g_pred = to_growth(P[:, h_idx], target_kind)
        g_true = to_growth(Y[:, h_idx], target_kind)
        price_pred = [P0]; price_true=[P0]
        for gp in g_pred: price_pred.append(price_pred[-1]*gp)
        for gt in g_true: price_true.append(price_true[-1]*gt)
        t = np.arange(len(price_true))
        plt.figure(figsize=(12,5))
        plt.plot(t, price_true, label="Actual (rolling h=1)", linestyle="--", alpha=0.8)
        plt.plot(t, price_pred, label="Pred (rolling h=1)")
        plt.title("Rolling h=1 Absolute Price Path")
        plt.xlabel("Test steps"); plt.ylabel("Price")
        plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()

if __name__ == "__main__":
    main()
