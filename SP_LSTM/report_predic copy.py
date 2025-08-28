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
import matplotlib.dates as mdates
import os
from old.DSConfig import DSConfig, FeatureConfig
from pandas.tseries.offsets import BDay
# from train_lstm import MIDataset, MultiInputLSTM

    # 1) 데이터/메타 로드

feature = FeatureConfig()


# ========== 설정 ==========
BATCH_SIZE = 128
ROLLING_H1_PLOT = True   # h=1 연속 경로 추가로 그릴지
# =========================

# def build_model_from_shapes(Xp, Xf, Xd, Y, device):
#     P = Xp.shape[-1]; F = Xf.shape[-1]; D = Xd.shape[-1]; H = Y.shape[-1]
#     m = MultiInputLSTM(P, F, D, hidden=96, layers=1, head_hidden=128, out_dim=H, dropout=0.2).to(device)
#     return m


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
# 멀티-호라이즌 (N,H) 예측/실제 행렬을 시계열 축에 정렬한 1D 시리즈 사전으로 변환하는 함수.

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


def report_predictions(model, test_loader, device, cfg):
    
# # ---------------- Load payload ----------------
#     get_dir = Path(cfg.dataset_dir)
#     payload_path = os.path.join(get_dir, f"{cfg.name}({cfg.code})_{cfg.end_date}.pkl")
#     payload = pd.read_pickle(payload_path)

#     meta = payload["meta"]
    feat = feature
#     lookback = meta["lookback"]
    horizons = cfg.horizons
    target_kind = cfg.target_kind
      
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
        close_idx = get_close_idx(feat)  

    # 5) 예측/정답 & base_closes & base_dates 수집 (배치 순서 그대로)
    Ys, Ps, base_closes_list = [], [], []
    base_dates_list = []   # ← 테스트 로더에서 날짜 수집

    with torch.no_grad():
        for b in test_loader:
            # 예측
            p = model(b["x_price"].to(device),
                    b["x_flow"].to(device),
                    b["x_fund"].to(device))
            Ps.append(p.cpu().numpy())
            Ys.append(b["y"].numpy())

            # 마지막 타임스텝 close 역스케일
            x_last = b["x_price"][:, -1, :].cpu().numpy()  # (B, P)
            if price_scaler is not None and hasattr(price_scaler, "inverse_transform"):
                try:
                    inv_all = price_scaler.inverse_transform(x_last)      # (B, P)
                    base_closes_list.append(inv_all[:, close_idx])        # (B,)
                except Exception:
                    inv_close = price_scaler.inverse_transform(x_last[:, close_idx:close_idx+1])
                    base_closes_list.append(inv_close.ravel())
            else:
                base_closes_list.append(x_last[:, close_idx])

            # ← 날짜 수집: MIDataset에서 ISO 문자열로 넘어오도록 해두었음
            if "base_date" in b:
                # b["base_date"]는 list[str] (DataLoader default_collate)
                base_dates_list.extend(b["base_date"])

    P = np.concatenate(Ps, axis=0)        # (N_pred, H)
    Y = np.concatenate(Ys, axis=0)        # (N_pred, H)
    base_closes = np.concatenate(base_closes_list, axis=0)  # (N_pred,)
    assert P.shape == Y.shape, "Pred/True shape mismatch"

    # ---- 날짜: test_loader에서 수집한 기준일을 그대로 사용 ----
    base_dates = pd.to_datetime(base_dates_list, errors="coerce", utc=True).tz_convert(None)
    if base_dates.isna().any():
        raise ValueError("base_date에 파싱되지 않는 값(NaT)이 있습니다.")

    # 길이 불일치 방어 (drop_last 등으로)
    N_pred = P.shape[0]
    if len(base_dates) != N_pred:
        N_eff = min(N_pred, len(base_dates))
        print(f"[WARN] 날짜/예측 길이 불일치: dates={len(base_dates)}, preds={N_pred} → {N_eff}로 맞춥니다.")
        P, Y, base_closes, base_dates = P[:N_eff], Y[:N_eff], base_closes[:N_eff], base_dates[:N_eff]
        N_pred = N_eff

    h_max = max(cfg.horizons)

    # x축 날짜: 마지막 기준일 뒤로 h_max 영업일까지 확장
    x_dates = base_dates if h_max == 0 else base_dates.append(
        pd.bdate_range(base_dates[-1] + BDay(1), periods=h_max)
    )
    # ----- 정렬 곡선 생성 -----
    P0 = float(base_closes[0])
    aligned_pred, aligned_true = build_aligned_series_from_base(
        preds=P, trues=Y, base_closes=base_closes,
        horizons=cfg.horizons, target_kind=cfg.target_kind
    )

    # ----- 플롯 (실제는 있는 구간만, 미래 구간은 예측만) -----
    # 필요 시 특정 horizon만 선택: use_h = [1,2,5]
    use_h = list(cfg.horizons)

    # x_dates 길이 부족하면 확장
    need_len = N_pred + max(use_h)
    if len(x_dates) < need_len:
        extra = pd.bdate_range(x_dates[-1] + BDay(1), periods=need_len - len(x_dates))
        x_dates = x_dates.append(extra)

    for h in use_h:
        y_true = aligned_true[h]   # 길이 = need_len, 미래 구간은 NaN일 수 있음
        y_pred = aligned_pred[h]

        m_true = ~np.isnan(y_true)  # 실제값 존재 구간
        m_pred = ~np.isnan(y_pred)  # 예측값 존재 구간

        plt.figure(figsize=(13, 5))
        # 예측: 전체 유효 구간
        plt.plot(x_dates[m_pred], y_pred[m_pred], marker='o', alpha=0.9, label=f"Pred +{h}d")
        # 실제: 존재 구간만 (미래 구간은 자동 미표시)
        if m_true.any():
            plt.plot(x_dates[m_true], y_true[m_true], marker='o', linestyle="--", alpha=0.7, label=f"True +{h}d")

        # x축 포맷
        ax = plt.gca()
        loc = mdates.AutoDateLocator()
        ax.xaxis.set_major_locator(loc)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
        plt.title(f"Absolute Price – +{h}B (Code={cfg.code}, target={cfg.target_kind})")
        plt.xlabel("Date"); plt.ylabel("Price")
        plt.grid(True); plt.legend(ncol=2); plt.tight_layout()

    # (선택) rolling h=1 도 날짜로 보고 싶다면:
    if ROLLING_H1_PLOT:
        t, price_true_roll, price_pred_roll = reconstruct_rolling_h1_path(
            preds=P, trues=Y, P0=P0, horizons=cfg.horizons, target_kind=cfg.target_kind
        )
        # t가 정수 인덱스라면 날짜로 치환 가능
        x_dates_roll = x_dates[t]
        plt.figure(figsize=(12, 5))
        plt.plot(x_dates_roll[:6], price_true_roll[:6], label="Actual (rolling h=1)",
                 marker='o', linestyle="--", alpha=0.8)
        plt.plot(x_dates_roll[:6], price_pred_roll[:6], label="Predicted (rolling h=1)",
                 marker='o', alpha=0.9)
        plt.title(f"[Rolling h=1] Absolute Price Path (Code={cfg.code}, target={cfg.target_kind})")
        plt.xlabel("Date"); plt.ylabel("Price")
        ax = plt.gca()
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
        plt.grid(True); plt.legend(); plt.tight_layout()

    plt.show()

