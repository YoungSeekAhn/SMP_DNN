# make_dataset.py
import os, pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from DSConfig import DSConfig, SplitConfig, FeatureConfig

split = SplitConfig()
# ---------------- Feature Config ----------------
feature = FeatureConfig()
    
# ---------------- Target ----------------
def build_target(df: pd.DataFrame, kind: str) -> np.ndarray:
    if kind == "logr":
        return np.log(df["close"]).diff().fillna(0.0).values
    elif kind == "pct":
        return df["close"].pct_change().fillna(0.0).values
    elif kind == "close":
        return df["close"].values
    else:
        raise ValueError("target_kind must be one of ['logr','pct','close']")


# ---------------- Split (time-ordered) ----------------
def time_split(df: pd.DataFrame, split: SplitConfig):
    n = len(df)
    i_tr = int(n * split.train_ratio)
    i_va = int(n * (split.train_ratio + split.val_ratio))
    tr = df.iloc[:i_tr].copy()
    va = df.iloc[i_tr:i_va].copy()
    te = df.iloc[i_va:].copy()
    return tr, va, te, i_tr, i_va

# ---------------- Scalers (fit on train only) ----------------
def fit_scalers(df: pd.DataFrame, feat_cfg):
    # 각 브랜치와 컬럼리스트를 '튜플'로 묶어서 순회
    branches = [
        ("price", feat_cfg.price_cols),
        ("flow",  feat_cfg.flow_cols),
        ("fund",  feat_cfg.fund_cols),
    ]

    scalers = {}
    for branch, cols in branches:
        if not cols:
            continue
        cols = [c for c in cols if c in df.columns]  # 존재하는 열만 사용
        if not cols:
            continue

        scaler = StandardScaler()
        scaler.fit(df[cols].astype(float))
        scalers[branch] = {"scaler": scaler, "cols": cols}

    return scalers

def apply_scalers(df: pd.DataFrame, scalers: dict) -> pd.DataFrame:
    out = df.copy()
    for branch, pack in scalers.items():
        cols   = pack["cols"]
        scaler = pack["scaler"]
        out[cols] = scaler.transform(out[cols].astype(float))
    return out
   
# ---------------- Make windows (inside each block only) ----------------
def make_windows_block(df, df_block: pd.DataFrame, y_block: np.ndarray,
                       lookback: int, horizons: Tuple[int, ...], feat_cfg):
    Hs = list(horizons); Hmax = max(Hs)
    Xp, Xf, Xd, Y = [], [], [], []
    
    p_candidates = feature.price_cols
    f_candidates = feature.flow_cols
    d_candidates = feature.fund_cols
        
    pcols = [c for c in p_candidates if c in df.columns]
    fcols = [c for c in f_candidates if c in df.columns]
    dcols = [c for c in d_candidates if c in df.columns]
    price = df_block[pcols].to_numpy()
    flow  = df_block[fcols].to_numpy()
    fund  = df_block[dcols].to_numpy()
    
    n = len(df_block)
    for t in range(lookback-1, n - Hmax):
        Xp.append(price[t-lookback+1:t+1])
        Xf.append(flow[t-lookback+1:t+1])
        Xd.append(fund[t-lookback+1:t+1])
        Y.append([y_block[t+h] for h in Hs])
    return (np.asarray(Xp, np.float32),
            np.asarray(Xf, np.float32),
            np.asarray(Xd, np.float32),
            np.asarray(Y,  np.float32))

def make_datasets(df, cfg, LOAD_CSV_FILE=True):
    
    if LOAD_CSV_FILE:
        # ---------------- Load Data from Saved CSV file----------------
        get_dir = Path(cfg.getdata_dir)
        filepath = os.path.join(get_dir, f"{cfg.name}({cfg.code})_{cfg.end_date}.csv")
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        print(f"Loaded data from: {filepath}")
        
    y_full = build_target(df, cfg.target_kind)
    
    tr, va, te, i_tr, i_va = time_split(df, split)
    y_tr = y_full[:i_tr]
    y_va = y_full[i_tr:i_va]
    y_te = y_full[i_va:]
    
    scalers = fit_scalers(tr, feature)
    tr_s = apply_scalers(tr, scalers)
    va_s = apply_scalers(va, scalers)
    te_s = apply_scalers(te, scalers)

    # ---------------- Save scalers ----------------
    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/scalers.pkl", "wb") as f:
        pickle.dump(scalers, f)
        
    # ---------------- Make windows (inside each block only) ----------------
    Xp_tr, Xf_tr, Xd_tr, Y_tr = make_windows_block(df, tr_s, y_tr, cfg.lookback, cfg.horizons, feature)
    Xp_va, Xf_va, Xd_va, Y_va = make_windows_block(df, va_s, y_va, cfg.lookback, cfg.horizons, feature)
    Xp_te, Xf_te, Xd_te, Y_te = make_windows_block(df, te_s, y_te, cfg.lookback, cfg.horizons, feature)

    # --- payload 딕셔너리 구성 ---
    payload = {
        # 학습 데이터
        "Xp_tr": Xp_tr, "Xf_tr": Xf_tr, "Xd_tr": Xd_tr, "Y_tr": Y_tr,
        # 검증 데이터
        "Xp_va": Xp_va, "Xf_va": Xf_va, "Xd_va": Xd_va, "Y_va": Y_va,
        # 테스트 데이터
        "Xp_te": Xp_te, "Xf_te": Xf_te, "Xd_te": Xd_te, "Y_te": Y_te,
        # 메타 정보 (cfg 등 포함)
        "meta": {
            "feature_config": feature,
            "lookback": cfg.lookback,
            "horizons": cfg.horizons,
            "target_kind": cfg.target_kind,
            "splits": {
                "train_ratio": split.train_ratio,
                "val_ratio": split.val_ratio,
                "test_ratio": split.test_ratio,
            },
            "lengths": {
                "train": len(Xp_tr),
                "val": len(Xp_va),
                "test": len(Xp_te),
            },
        }
    }

    # ---------------- Save payload ----------------
    get_dir = Path(cfg.dataset_dir)
    get_dir.mkdir(exist_ok=True, parents=True)
    dataset_path = os.path.join(get_dir, f"{cfg.name}({cfg.code})_{cfg.end_date}.pkl")

    with open(dataset_path, "wb") as f:
        pickle.dump(payload, f)

    print("Metadata:",payload["meta"])
    
    return payload