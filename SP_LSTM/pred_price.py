# plot_test_prices_aligned.py
# ------------------------------------------------------------
# 저장된 모델과 스케일러를 불러와
# 1) horizon 정렬(Aligned) 절대가격 곡선: h=1,2,5 ... (예측 vs 실제)
# 2) (선택) rolling h=1 경로
# 를 그려주는 완전 실행 스크립트
# ------------------------------------------------------------
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os
from DSConfig import DSConfig
from train_dataset_lstm import MIDataset, MultiInputLSTM

    # 1) 데이터/메타 로드
cfg = DSConfig()

# ========== 설정 ==========
BATCH_SIZE = 128
ROLLING_H1_PLOT = True   # h=1 연속 경로 추가로 그릴지
# =========================

def build_model_from_shapes(Xp, Xf, Xd, Y, device):
    P = Xp.shape[-1]; F = Xf.shape[-1]; D = Xd.shape[-1]; H = Y.shape[-1]
    m = MultiInputLSTM(P, F, D, hidden=96, layers=1, head_hidden=128, out_dim=H, dropout=0.2).to(device)
    return m


# ===== 헬퍼 =====
def get_close_idx(feat):
        # price_cols 가져오기
    if isinstance(feat, dict):
        cols = feat.get("price_cols", [])
    else:
        cols = getattr(feat, "price_cols", [])

    return cols.index("close") if "close" in cols else 3

def to_growth(arr: np.ndarray, target_kind: str) -> np.ndarray:
    if target_kind == "logr":
        return np.exp(arr)
    elif target_kind == "pct":
        return 1.0 + arr
    elif target_kind == "close":
        # close는 절대가격 자체이므로 성장률 개념이 아님 → None 반환하여 분기별 처리
        return None
    else:
        raise ValueError("target_kind must be in {'logr','pct','close'}")


# ===== horizon 정렬 시계열 (예측/실제 모두 생성) =====
def build_aligned_series_from_base(
    preds: np.ndarray,     # (N, H)
    trues: np.ndarray,     # (N, H)
    base_closes: np.ndarray,   # (N,)
    horizons: Union[List[int], Tuple[int, ...]],
    target_kind: str
):
    if not isinstance(horizons, (list, tuple)):
        horizons = [int(horizons)]
    Hs = list(horizons)
    h_max = max(Hs)
    N, H = preds.shape
    assert trues.shape == preds.shape, "preds and trues must have same shape"

    aligned_pred = {h: np.full(N + h_max, np.nan, dtype=float) for h in Hs}
    aligned_true = {h: np.full(N + h_max, np.nan, dtype=float) for h in Hs}

    if target_kind in ("logr", "pct"):
        g_pred = to_growth(preds, target_kind)   # (N, H)
        g_true = to_growth(trues, target_kind)   # (N, H)

        for j, h in enumerate(Hs):
            for k in range(N):
                t = k + h
                if t < N + h_max:
                    aligned_pred[h][t] = base_closes[k] * g_pred[k, j]
                    aligned_true[h][t] = base_closes[k] * g_true[k, j]
    else:
        # close 절대가격: preds/trues 자체가 절대가라고 간주
        for j, h in enumerate(Hs):
            for k in range(N):
                t = k + h
                if t < N + h_max:
                    aligned_pred[h][t] = preds[k, j]
                    aligned_true[h][t] = trues[k, j]

    return aligned_pred, aligned_true


# ===== rolling h=1 (연속 one-step 경로) =====
def reconstruct_rolling_h1_path(
    preds: np.ndarray,     # (N, H)
    trues: np.ndarray,     # (N, H)
    P0: float,
    horizons: Union[List[int], Tuple[int, ...]],
    target_kind: str
):
    if not isinstance(horizons, (list, tuple)):
        horizons = [int(horizons)]
    h_idx = horizons.index(1) if 1 in horizons else 0

    if target_kind in ("logr", "pct"):
        r_pred = preds[:, h_idx]
        r_true = trues[:, h_idx]
        g_pred = to_growth(r_pred, target_kind) if r_pred.ndim == 1 else to_growth(r_pred.squeeze(), target_kind)
        g_true = to_growth(r_true, target_kind) if r_true.ndim == 1 else to_growth(r_true.squeeze(), target_kind)

        price_pred = [P0]; price_true = [P0]
        for gp in g_pred: price_pred.append(price_pred[-1] * gp)
        for gt in g_true: price_true.append(price_true[-1] * gt)
        t = np.arange(len(price_true))
        return t, np.array(price_true), np.array(price_pred)

    elif target_kind == "close":
        # close 직접 예측이라면 성장률 누적이 아닌 절대가 → rolling의 의미가 희미
        # 여기서는 간단히: 첫 P0에서 시작, 이후 예측/실제 close를 연결(보정 없음)
        price_true = np.concatenate([[P0], trues[:, h_idx]])
        price_pred = np.concatenate([[P0], preds[:, h_idx]])
        t = np.arange(len(price_true))
        return t, price_true, price_pred

    else:
        raise ValueError("invalid target_kind")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- Load payload ----------------
    get_dir = Path(cfg.dataset_dir)
    payload_path = os.path.join(get_dir, f"{cfg.name}({cfg.code})_{cfg.end_date}.pkl")
    payload = pd.read_pickle(payload_path)

    meta = payload["meta"]
    feat = meta["feature_config"]
    lookback = meta["lookback"]
    horizons = meta["horizons"]
    target_kind = meta["target_kind"]


    # 2) Test loader 구성
    ds_te = MIDataset(payload["Xp_te"], payload["Xf_te"], payload["Xd_te"], payload["Y_te"])
    test_loader = DataLoader(ds_te, batch_size=BATCH_SIZE, shuffle=False)

    # 3) 모델 복원
    model = build_model_from_shapes(payload["Xp_tr"], payload["Xf_tr"], payload["Xd_tr"], payload["Y_tr"], device)
    model_dir = Path(cfg.model_dir)
    model_path = os.path.join(model_dir, f"{cfg.name}({cfg.code})_{cfg.end_date}.pt")
    assert os.path.exists(model_path), f"[ERR] model not found: {model_path}"
    print(f"Loading model from: {model_path}")
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt["model"]); model.eval()
    target_kind = ckpt.get("target_kind", target_kind)  # ckpt 우선

    # 4) 스케일러 로드 (price 역스케일)
    scalers_path = Path("artifacts") / "scalers.pkl"
    assert scalers_path.exists(), f"[ERR] scalers not found: {scalers_path}"
    with open(scalers_path, "rb") as f:
        scalers = pickle.load(f)
    # 브랜치 스케일러/컬럼 추출
    price_pack   = scalers.get("price", {})
    price_scaler = price_pack.get("scaler", None)
    price_cols   = list(price_pack.get("cols", []))
    
    # close 인덱스 (스케일러에 저장된 cols 우선, 없으면 feat에서)
    if price_cols:
        close_idx = price_cols.index("close") if "close" in price_cols else 3
    else:
        close_idx = get_close_idx(feat)  # 당신이 이미 만든 함수

    # 5) 예측/정답 & base_closes 수집 (배치 순서 그대로)
    Ys, Ps, base_closes_list = [], [], []
    with torch.no_grad():
        for b in test_loader:
            # 예측
            p = model(b["x_price"].to(device),
                      b["x_flow"].to(device),
                      b["x_fund"].to(device))
            Ps.append(p.cpu().numpy())
            Ys.append(b["y"].numpy())

            # 전일 종가(입력 마지막 close) 역스케일 수집
            x_last = b["x_price"][:, -1, :].cpu().numpy()  # (B, P)
            
            if price_scaler is not None and hasattr(price_scaler, "inverse_transform"):
                # 전체 역변환이 가장 안전 (컬럼 순서가 fit과 동일해야 함)
                try:
                    inv_all = price_scaler.inverse_transform(x_last)      # (B, P)
                    base_closes_list.append(inv_all[:, close_idx])        # (B,)
                except Exception:
                    # 실패 시 close 단일 컬럼만 역변환 시도
                    inv_close = price_scaler.inverse_transform(x_last[:, close_idx:close_idx+1])
                    base_closes_list.append(inv_close.ravel())
            else:
                # 스케일러 없으면 스케일된 값을 그대로 사용 (주의)
                base_closes_list.append(x_last[:, close_idx])

    P = np.concatenate(Ps, axis=0)        # (N, H)
    Y = np.concatenate(Ys, axis=0)        # (N, H)
    base_closes = np.concatenate(base_closes_list, axis=0)  # (N,)
    assert P.shape == Y.shape, "Pred/True shape mismatch"

    # 6) 초기 P0 (rolling h=1용; 첫 배치 첫 샘플의 전일 close)
    P0 = float(base_closes[0])
    print(f"P0 (initial close): {P0:.4f} | target_kind={target_kind} | horizons={horizons}")

    # 7) 정렬된(Aligned) 절대가격 곡선 (예측 vs 실제)
    aligned_pred, aligned_true = build_aligned_series_from_base(
        preds=P, trues=Y, base_closes=base_closes,
        horizons=horizons, target_kind=target_kind
    )

   
    for h in horizons:
        plt.figure(figsize=(13, 5))
        plt.plot(aligned_true[h], marker='o', linestyle="--", alpha=0.7, label=f"True h={h}")
        plt.plot(aligned_pred[h], marker='o', alpha=0.9, label=f"Pred h={h}")
        plt.title(f"[Aligned] Absolute Price by Horizon (Code={cfg.code}, target={target_kind})")
        plt.xlabel("Test timeline index")
        plt.ylabel("Price")
        plt.grid(True); plt.legend(ncol=2); plt.tight_layout()


    # 8) (선택) rolling h=1 경로
    if ROLLING_H1_PLOT:
        t, price_true_roll, price_pred_roll = reconstruct_rolling_h1_path(
            preds=P, trues=Y, P0=P0, horizons=horizons, target_kind=target_kind
        )
        plt.figure(figsize=(12, 5))
        plt.plot(t[:6], price_true_roll[:6], label="Actual (rolling h=1)", marker = 'o', linestyle="--", alpha=0.8)
        plt.plot(t[:6], price_pred_roll[:6], label="Predicted (rolling h=1)", marker = 'o', alpha=0.9)
        plt.title(f"[Rolling h=1] Absolute Price Path (Code={cfg.code}, target={target_kind})")
        plt.xlabel("Test steps"); plt.ylabel("Price")
        plt.grid(True); plt.legend(); plt.tight_layout()
        
    plt.show()


if __name__ == "__main__":
    main()
