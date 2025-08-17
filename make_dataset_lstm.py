# make_dataset.py
import os, pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ---------------- Config ----------------
@dataclass
class SplitConfig:
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2


@dataclass
class DSConfig:
    code: str = "005930"
    lookback: int = 30
    horizons: Tuple[int, ...] = (1, 2, 5)
    target_kind: str = "logr"  # "logr" | "pct" | "close"

split = SplitConfig()
cfg = DSConfig()

out_dir = Path.cwd()
filepath = os.path.join(out_dir, f"{cfg.code}_merged_data.csv")
df = pd.read_csv(filepath, index_col=0, parse_dates=True)

# ---------------- Feature Config ----------------
feature_config = {
    "price_cols": ["open", "high", "low", "close", "volume", "chg_pct"],
    "flow_cols":  ["inst_sum", "inst_ext", "retail", "foreign"],
    "fund_cols":  ["per", "pbr", "div", "bps", "eps", "dps"],
}

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

y_full = build_target(df, cfg.target_kind)

# ---------------- Split (time-ordered) ----------------
def time_split(df: pd.DataFrame, split: SplitConfig):
    n = len(df)
    i_tr = int(n * split.train_ratio)
    i_va = int(n * (split.train_ratio + split.val_ratio))
    tr = df.iloc[:i_tr].copy()
    va = df.iloc[i_tr:i_va].copy()
    te = df.iloc[i_va:].copy()
    return tr, va, te, i_tr, i_va

tr, va, te, i_tr, i_va = time_split(df, split)
y_tr = y_full[:i_tr]
y_va = y_full[i_tr:i_va]
y_te = y_full[i_va:]

# ---------------- Scalers (fit on train only) ----------------
def fit_scalers(df_block: pd.DataFrame, feat_cfg: Dict[str, List[str]]):
    scalers = {}
    for branch, cols_key in [("price","price_cols"), ("flow","flow_cols"), ("fund","fund_cols")]:
        cols = feat_cfg.get(cols_key, [])
        if cols:
            scalers[branch] = StandardScaler().fit(df_block[cols].values)
    return scalers

def apply_scalers(df_block: pd.DataFrame, scalers, feat_cfg):
    out = df_block.copy()
    for branch, cols_key in [("price","price_cols"), ("flow","flow_cols"), ("fund","fund_cols")]:
        cols = feat_cfg.get(cols_key, [])
        if cols and branch in scalers:
            out.loc[:, cols] = scalers[branch].transform(out[cols].values)
    return out

scalers = fit_scalers(tr, feature_config)
tr_s = apply_scalers(tr, scalers, feature_config)
va_s = apply_scalers(va, scalers, feature_config)
te_s = apply_scalers(te, scalers, feature_config)

Path("artifacts").mkdir(exist_ok=True, parents=True)
with open("artifacts/scalers.pkl", "wb") as f:
    pickle.dump(scalers, f)

# ---------------- Make windows (inside each block only) ----------------
def make_windows_block(df_block: pd.DataFrame, y_block: np.ndarray,
                       lookback: int, horizons: Tuple[int, ...], feat_cfg):
    Hs = list(horizons); Hmax = max(Hs)
    Xp, Xf, Xd, Y = [], [], [], []
    price = df_block[feat_cfg["price_cols"]].values
    flow  = df_block[feat_cfg["flow_cols"]].values
    fund  = df_block[feat_cfg["fund_cols"]].values
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

Xp_tr, Xf_tr, Xd_tr, Y_tr = make_windows_block(tr_s, y_tr, cfg.lookback, cfg.horizons, feature_config)
Xp_va, Xf_va, Xd_va, Y_va = make_windows_block(va_s, y_va, cfg.lookback, cfg.horizons, feature_config)
Xp_te, Xf_te, Xd_te, Y_te = make_windows_block(te_s, y_te, cfg.lookback, cfg.horizons, feature_config)

# ---------------- Save payload ----------------
Path("datasets").mkdir(exist_ok=True, parents=True)
payload = {
    "Xp_tr": Xp_tr, "Xf_tr": Xf_tr, "Xd_tr": Xd_tr, "Y_tr": Y_tr,
    "Xp_va": Xp_va, "Xf_va": Xf_va, "Xd_va": Xd_va, "Y_va": Y_va,
    "Xp_te": Xp_te, "Xf_te": Xf_te, "Xd_te": Xd_te, "Y_te": Y_te,
    "meta": {
        "Code": cfg.code,
        "window": {"lookback": cfg.lookback, "horizons": list(cfg.horizons)},
        "feature_config": feature_config,
        "target_kind": cfg.target_kind,
        "splits": {"train_ratio": split.train_ratio, "val_ratio": split.val_ratio, "test_ratio": split.test_ratio},
        "scalers": "StandardScaler per-branch (fit on train)",
    }
}
out_path = Path("datasets") / f"{cfg.code}_dataset.pkl"
pd.to_pickle(payload, out_path)
print(f"Saved dataset payload -> {out_path}")
