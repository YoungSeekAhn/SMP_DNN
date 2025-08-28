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

# ------------------------------------------------------------
#  목표 날짜 기준 테이블 생성: date, true, pred_h1, pred_h2, ...
#  (길이 불일치 자동 보정)
# ------------------------------------------------------------
def build_target_aligned_table(
    aligned_pred: dict[int, np.ndarray],
    aligned_true: dict[int, np.ndarray],
    x_dates,                      # DatetimeIndex/array-like
    horizons                      # 예: [1,2,5]
):
    x_idx = pd.DatetimeIndex(pd.to_datetime(x_dates, errors="coerce")).tz_localize(None)
    if x_idx.isna().any():
        raise ValueError("x_dates에 NaT가 포함되어 있습니다.")
    T = len(x_idx)
    df = pd.DataFrame({"date": x_idx})
    use_h = [int(h) for h in horizons if h in aligned_pred]

    def _fit_len(arr, T):
        a = np.asarray(arr, dtype=float)
        if len(a) == T: return a
        if len(a) > T:  return a[:T]
        out = np.full(T, np.nan, dtype=float)
        out[:len(a)] = a
        return out

    # 예측 열: 이미 목표 날짜 위치에 정렬되어 있으므로 그대로 사용
    for h in use_h:
        df[f"pred_h{h}"] = _fit_len(aligned_pred[h], T)

    # true: 해당 날짜의 실제 절대가(하나의 시계열) — h가 작은 것부터 우선 채움
    true_series = np.full(T, np.nan, dtype=float)
    for h in sorted(use_h):
        arr = _fit_len(aligned_true[h], T)
        m = np.isnan(true_series) & np.isfinite(arr)
        true_series[m] = arr[m]
    df["true"] = true_series

    return df

def extend_x_dates(base_dates, aligned_pred: dict, horizons):
    # 1) 날짜 표준화 & 검증
    x = pd.DatetimeIndex(pd.to_datetime(base_dates, errors="coerce")).tz_localize(None)
    if len(x) == 0 or x.isna().any():
        raise ValueError("base_dates가 비었거나 NaT가 포함되어 있습니다.")
    # (선택) 정렬 보장
    if not x.is_monotonic_increasing:
        x = x.sort_values()

    # 2) need_len 계산 (aligned_pred에 실제로 존재하는 h만 사용)
    use_h = [int(h) for h in horizons if h in aligned_pred]
    if not use_h:
        raise ValueError(f"aligned_pred에 {list(horizons)} 중 해당하는 키가 없습니다. keys={list(aligned_pred.keys())}")
    need_len = max(len(aligned_pred[h]) for h in use_h)

    # 3) 확장 없으면 그대로 반환
    if len(x) >= need_len:
        return x

    # 4) 영업일 기준으로 need_len까지 확장
    add = need_len - len(x)

    # 단순 주말 제외(공휴일 미반영)
    extra = pd.bdate_range(x[-1] + BDay(1), periods=add)

    return x.append(extra)


# ------------------------------------------------------------
#  보고 + 플롯: true는 해당 날짜, pred는 예측 목표 날짜에 표시
# ------------------------------------------------------------
def report_predictions(model, test_loader, device, cfg):
    model.eval()
    feat = feature                 # 외부의 FeatureConfig 객체를 쓰는 전제
    horizons = list(cfg.horizons)
    target_kind = cfg.target_kind

    # --- 스케일러 로드 & close_idx ---
    scalers_path = Path("artifacts") / "scalers.pkl"
    assert scalers_path.exists(), f"[ERR] scalers not found: {scalers_path}"
    with open(scalers_path, "rb") as f:
        scalers = pickle.load(f)
    price_pack   = scalers.get("price", {})
    price_scaler = price_pack.get("scaler", None)
    price_cols   = list(price_pack.get("cols", []))
    if price_cols:
        close_idx = price_cols.index("close") if "close" in price_cols else 3
    else:
        close_idx = get_close_idx(feat)

    # --- 예측/정답/기준가/날짜 수집 ---
    Ps, Ys, base_closes_list, base_dates_list = [], [], [], []

    with torch.no_grad():
        for b in test_loader:
            p = model(b["x_price"].to(device),
                      b["x_flow"].to(device),
                      b["x_fund"].to(device),
                      b["x_glob"].to(device)
                      ).cpu()
            Ps.append(p.cpu().numpy())
            Ys.append(b["y"].numpy())

            x_last = b["x_price"][:, -1, :].cpu().numpy()  # (B, P)
            if price_scaler is not None and hasattr(price_scaler, "inverse_transform"):
                try:
                    inv_all = price_scaler.inverse_transform(x_last)      # (B, P)
                    close_vec = inv_all[:, close_idx]
                except Exception:
                    # 폴백: close 단일 컬럼만 역변환
                    close_only = x_last[:, close_idx:close_idx+1]
                    try:
                        close_vec = price_scaler.inverse_transform(close_only).ravel()
                    except Exception:
                        print("[WARN] inverse_transform 실패. 스케일된 close를 임시 사용합니다.")
                        close_vec = x_last[:, close_idx]
            else:
                close_vec = x_last[:, close_idx]
            base_closes_list.append(close_vec)

            if "base_date" in b:  # MIDataset가 ISO 문자열로 제공
                base_dates_list.extend(b["base_date"])

    P = np.concatenate(Ps, axis=0)                 # (N, H)
    Y = np.concatenate(Ys, axis=0)                 # (N, H)
    base_closes = np.concatenate(base_closes_list) # (N,)
    assert P.shape == Y.shape, "Pred/True shape mismatch"

    # --- 기준일 확보 ---
    base_dates = pd.to_datetime(base_dates_list, errors="coerce", utc=True).tz_convert(None)
    if base_dates.isna().any() or len(base_dates) == 0:
        raise RuntimeError("기준일(base_dates)을 수집하지 못했습니다. Dataset이 base_date를 반환하도록 하세요.")
    if len(base_dates) != len(P):
        raise RuntimeError(f"날짜/예측 길이 불일치: dates={len(base_dates)}, preds={len(P)}. "
                           f"DataLoader(shuffle=False, drop_last=False) 및 Dataset 날짜 반환을 확인하세요.")

    # --- 정렬 배열 생성 ---
    aligned_pred, aligned_true = build_aligned_series_from_base(
        preds=P, trues=Y, base_closes=base_closes,
        horizons=horizons, target_kind=target_kind
    )

    # --- x_dates: (N + h_max)까지 영업일 확장 ---
    N = len(base_dates)
   
    x_dates = extend_x_dates(base_dates, aligned_pred, horizons)
    
    # --- 마지막 목표일 값이 비었으면 보정(선택) ---
    def _to_abs(val, kind, base):
        if kind == "pct":  return float(base) * (1.0 + float(val))
        if kind == "logr": return float(base) * float(np.exp(val))
        if kind == "close":return float(val)
        raise ValueError(kind)

    for h in horizons:
        if h not in aligned_pred: continue
        j = horizons.index(h)             # preds/trues의 열 인덱스
        t_last = (N - 1) + h              # 마지막 기준일 + h일
        if t_last < len(aligned_pred[h]) and not np.isfinite(aligned_pred[h][t_last]):
            if np.isfinite(P[-1, j]) and np.isfinite(base_closes[-1]):
                aligned_pred[h][t_last] = _to_abs(P[-1, j], target_kind, base_closes[-1])

    # --- 날짜별 한 줄 테이블 생성 (true는 해당 날짜, pred는 목표 날짜) ---
    table = build_target_aligned_table(
        aligned_pred=aligned_pred,
        aligned_true=aligned_true,
        x_dates=x_dates,
        horizons=horizons
    )
    x_idx = pd.DatetimeIndex(table["date"])
    last_base = pd.Timestamp(base_dates[-1])
    def _to_abs(val, kind, base):
        if kind == "pct":  return float(base) * (1.0 + float(val))
        if kind == "logr": return float(base) * float(np.exp(val))
        if kind == "close":return float(val)
        raise ValueError(kind)

    for h in horizons:
        j = horizons.index(h)                # P[:, j]에 해당
        target_day = last_base + BDay(h)     # 마지막 실제일 + h영업일
        if target_day in x_idx:
            i = x_idx.get_loc(target_day)
            col = f"pred_h{h}"
            # pred 값이 NaN이면 마지막 샘플로부터 보정해 채움
            if pd.isna(table.at[i, col]) and np.isfinite(P[-1, j]) and np.isfinite(base_closes[-1]):
                table.at[i, col] = _to_abs(P[-1, j], target_kind, base_closes[-1])
                
            
    # (옵션) 기존 포맷 유지하려면 true_h{h} 추가
    for h in horizons:
        table[f"true_h{h}"] = table["true"]

    # --- 저장 ---
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"pred_true_wide_{cfg.code}.csv"
    # 보기 좋은 열 순서
    ordered_cols = ["date", "true"] + [f"pred_h{h}" for h in horizons] + [f"true_h{h}" for h in horizons]
    table[ordered_cols].to_csv(out_csv, index=False)
    print(f"[saved] {out_csv}")

    # --- 한 장에 여러 h를 그리던 기존 블록을 아래로 교체 ---
    dfp = table.copy()
    dfp["date"] = pd.to_datetime(dfp["date"], errors="coerce")

    for h in horizons:
        pred_col = f"pred_h{h}"
        if pred_col not in dfp.columns:
            continue

        # 값 준비 (float로 통일)
        y_true = dfp["true"].astype(float).to_numpy()
        y_pred = dfp[pred_col].astype(float).to_numpy()
        x      = dfp["date"].to_numpy()

        # 유효 구간(둘 중 하나라도 값이 있음)
        m = np.isfinite(y_true) | np.isfinite(y_pred)
        if not m.any():
            continue

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(x[m], y_true[m], label="True",marker='o', linestyle='--', alpha=0.9, linewidth=2)
        ax.plot(x[m], y_pred[m], marker='o', linestyle='-', alpha=0.9, label=f"Pred +{h}d")

        # x축 날짜 포맷
        loc = mdates.AutoDateLocator()
        ax.xaxis.set_major_locator(loc)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))

        # y축 포맷(정수/1자리 원하면 주석 해제)
        # ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))  # 정수
        # ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))  # 소수점 1자리

        ax.set_title(f"True vs Pred (+{h}d) — {cfg.name}({cfg.code})")
        ax.set_xlabel("Date"); ax.set_ylabel("Price")
        ax.grid(True); ax.legend()
        plt.tight_layout()
    plt.show()

