import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from sklearn.preprocessing import StandardScaler

# --- Optional: PyTorch dataset (you can remove if not needed) ---
try:
    import torch
    from torch.utils.data import Dataset
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
    class Dataset:  # dummy
        pass

# =============================================================
# 1) Investor subcategory aggregator
# =============================================================

INSTITUTION_PARTS_KO = [
    "금융투자","보험","투신","사모","은행","기타금융","연기금","기타법인"
]

def aggregate_investor(value_df: pd.DataFrame) -> pd.DataFrame:
    \"\"\"
    Takes a PyKRX 'get_market_trading_value_by_date(..., detail=True)' style DataFrame:
      columns include subcategories like 금융투자, 보험, 투신, 사모, 은행, 기타금융, 연기금, 기타법인, 개인, 외국인, 기타외국인, 전체
    Returns a DataFrame with columns: retail, foreign, institution
      retail      = 개인
      foreign     = 외국인 + 기타외국인 (if exist)
      institution = sum of institution subcategories
    Index must be DatetimeIndex (if not, will try to convert '날짜'/'date')
    \"\"\"
    if value_df is None or len(value_df) == 0:
        raise ValueError("Empty investor dataframe")

    df = value_df.copy()

    # ensure datetime index named 'date'
    if not isinstance(df.index, pd.DatetimeIndex):
        if "날짜" in df.columns:
            df["날짜"] = pd.to_datetime(df["날짜"])
            df = df.set_index("날짜")
        elif "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        else:
            try:
                df.index = pd.to_datetime(df.index)
            except Exception as e:
                raise ValueError("Provide DatetimeIndex or '날짜'/'date' column.") from e
    df.index.name = "date"
    df = df.sort_index()

    # Numeric conversion
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # retail
    retail = df["개인"] if "개인" in df.columns else df.get("retail")
    if retail is None:
        raise ValueError("Cannot find '개인' column for retail.")

    # foreign (외국인 + 기타외국인 if available)
    foreign_cols = []
    if "외국인" in df.columns:
        foreign_cols.append("외국인")
    if "기타외국인" in df.columns:
        foreign_cols.append("기타외국인")
    if not foreign_cols and "foreign" in df.columns:
        foreign_series = df["foreign"]
    else:
        foreign_series = df[foreign_cols].sum(axis=1)

    # institution sum of parts
    exist_parts = [c for c in INSTITUTION_PARTS_KO if c in df.columns]
    if exist_parts:
        institution = df[exist_parts].sum(axis=1)
    else:
        # fallback
        if "기관합계" in df.columns:
            institution = df["기관합계"]
        elif "institution" in df.columns:
            institution = df["institution"]
        else:
            raise ValueError("Cannot build 'institution': subcategory parts or '기관합계' not found.")

    out = pd.DataFrame({
        "retail": retail,
        "foreign": foreign_series,
        "institution": institution
    })
    # Replace NaN with 0 (no trades recorded for that category)
    out = out.fillna(0.0)
    return out

# =============================================================
# 2) Standardize OHLCV & Fundamental
# =============================================================

def standardize_ohlcv(ohlcv: pd.DataFrame) -> pd.DataFrame:
    if ohlcv is None or len(ohlcv) == 0:
        raise ValueError("Empty OHLCV dataframe")
    df = ohlcv.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        for c in ["날짜","date","Date"]:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c])
                df = df.set_index(c)
                break
        else:
            df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    df = df.sort_index()
    rename = {"시가":"open","고가":"high","저가":"low","종가":"close","거래량":"volume","거래대금":"value","등락률":"chg_pct"}
    df = df.rename(columns=rename)
    keep = [c for c in ["open","high","low","close","volume","value","chg_pct"] if c in df.columns]
    df = df[keep].apply(pd.to_numeric, errors="coerce")
    # drop rows without close
    df = df[df["close"].notna()]
    return df

def standardize_fundamental(fund: pd.DataFrame) -> pd.DataFrame:
    if fund is None or len(fund) == 0:
        raise ValueError("Empty Fundamental dataframe")
    df = fund.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        for c in ["날짜","date","Date"]:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c])
                df = df.set_index(c)
                break
        else:
            df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    df = df.sort_index()
    rename = {"PER":"per","PBR":"pbr","DIV":"div","BPS":"bps","EPS":"eps","DPS":"dps"}
    df = df.rename(columns=rename)
    keep = [c for c in ["per","pbr","div","bps","eps","dps"] if c in df.columns]
    df = df[keep].apply(pd.to_numeric, errors="coerce")
    # monthly/quarterly sparse -> ffill
    df = df.ffill()
    return df

# =============================================================
# 3) Merge, target, splits, scaling, windows
# =============================================================

@dataclass
class SplitConfig:
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

@dataclass
class WindowConfig:
    lookback: int = 30
    horizon: int = 1

@dataclass
class FeatureConfig:
    price_cols: List[str]
    flow_cols: List[str]
    fund_cols: List[str]

def merge_sources(ohlcv_df: pd.DataFrame, flow_df: pd.DataFrame, fund_df: pd.DataFrame) -> pd.DataFrame:
    out = standardize_ohlcv(ohlcv_df)
    flow = aggregate_investor(flow_df)
    out = out.join(flow, how="outer")
    if fund_df is not None and len(fund_df):
        fund = standardize_fundamental(fund_df)
        out = out.join(fund, how="outer")
    out = out.sort_index()
    # basic cleanup
    out = out.replace([np.inf,-np.inf], np.nan)
    out = out.dropna(subset=["close"])
    # investor NaN -> 0
    for c in ["retail","foreign","institution"]:
        if c in out.columns:
            out[c] = out[c].fillna(0.0)
    return out

def add_targets(df: pd.DataFrame, horizon: int = 1, target_kind: str = "logr") -> pd.DataFrame:
    out = df.copy()
    fut = out["close"].shift(-horizon)
    if target_kind == "logr":
        out["target"] = np.log(fut / out["close"])
    elif target_kind == "close":
        out["target"] = fut
    elif target_kind == "direction":
        out["target"] = (np.log(fut / out["close"]) > 0).astype(float)
    else:
        raise ValueError("target_kind must be 'logr'|'close'|'direction'")
    out = out.dropna(subset=["target"])
    return out

def time_split(df: pd.DataFrame, split: SplitConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    i_tr = int(n * split.train_ratio)
    i_va = int(n * (split.train_ratio + split.val_ratio))
    tr = df.iloc[:i_tr].copy()
    va = df.iloc[i_tr:i_va].copy()
    te = df.iloc[i_va:].copy()
    return tr, va, te

class BranchScalers:
    def __init__(self):
        self.price = StandardScaler()
        self.flow  = StandardScaler()
        self.fund  = StandardScaler()

def fit_scalers(tr: pd.DataFrame, feat: FeatureConfig) -> BranchScalers:
    sc = BranchScalers()
    if feat.price_cols: sc.price.fit(tr[feat.price_cols].values)
    if feat.flow_cols:  sc.flow.fit(tr[feat.flow_cols].values)
    if feat.fund_cols:  sc.fund.fit(tr[feat.fund_cols].values)
    return sc

def apply_scalers(df: pd.DataFrame, sc: BranchScalers, feat: FeatureConfig) -> pd.DataFrame:
    out = df.copy()
    if feat.price_cols: out.loc[:, feat.price_cols] = sc.price.transform(out[feat.price_cols].values)
    if feat.flow_cols:  out.loc[:, feat.flow_cols]  = sc.flow.transform(out[feat.flow_cols].values)
    if feat.fund_cols:  out.loc[:, feat.fund_cols]  = sc.fund.transform(out[feat.fund_cols].values)
    return out

def build_windows(df: pd.DataFrame, feat: FeatureConfig, lookback: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    P = len(feat.price_cols) if feat.price_cols else 0
    Q = len(feat.flow_cols) if feat.flow_cols else 0
    R = len(feat.fund_cols) if feat.fund_cols else 0
    Xp, Xf, Xd, y = [], [], [], []
    vals_price = df[feat.price_cols].values if P else None
    vals_flow  = df[feat.flow_cols].values if Q else None
    vals_fund  = df[feat.fund_cols].values if R else None
    targ = df["target"].values
    N = len(df)
    L = lookback
    for t in range(L-1, N):
        xp = vals_price[t-L+1:t+1] if P else np.zeros((L,0), dtype=np.float32)
        xf = vals_flow[t-L+1:t+1]  if Q else np.zeros((L,0), dtype=np.float32)
        xd = vals_fund[t-L+1:t+1]  if R else np.zeros((L,0), dtype=np.float32)
        yt = targ[t]
        if np.isnan(yt): continue
        if P and np.isnan(xp).any(): continue
        if Q and np.isnan(xf).any(): continue
        if R and np.isnan(xd).any(): continue
        Xp.append(xp); Xf.append(xf); Xd.append(xd); y.append(yt)
    return (np.asarray(Xp, dtype=np.float32),
            np.asarray(Xf, dtype=np.float32),
            np.asarray(Xd, dtype=np.float32),
            np.asarray(y,  dtype=np.float32))

# Optional PyTorch dataset
if TORCH_AVAILABLE:
    class MultiInputTSDataset(Dataset):
        def __init__(self, Xp, Xf, Xd, y):
            self.Xp = torch.from_numpy(Xp)
            self.Xf = torch.from_numpy(Xf)
            self.Xd = torch.from_numpy(Xd)
            self.y  = torch.from_numpy(y).view(-1,1)
        def __len__(self): return len(self.y)
        def __getitem__(self, i):
            return {"x_price": self.Xp[i], "x_flow": self.Xf[i], "x_fund": self.Xd[i], "y": self.y[i]}

# =============================================================
# 4) High-level convenience
# =============================================================

@dataclass
class PipelineConfig:
    lookback: int = 30
    horizon: int = 1
    target_kind: str = "logr"
    split: SplitConfig = SplitConfig()
    price_cols: Optional[List[str]] = None
    flow_cols: Optional[List[str]]  = None
    fund_cols: Optional[List[str]]  = None

def auto_feature_config(df: pd.DataFrame, cfg: PipelineConfig) -> FeatureConfig:
    p_candidates = ["open","high","low","close","volume","value","chg_pct"]
    f_candidates = ["retail","foreign","institution"]
    d_candidates = ["per","pbr","div","bps","eps","dps"]
    pcols = cfg.price_cols if cfg.price_cols is not None else [c for c in p_candidates if c in df.columns]
    fcols = cfg.flow_cols  if cfg.flow_cols  is not None else [c for c in f_candidates if c in df.columns]
    dcols = cfg.fund_cols  if cfg.fund_cols  is not None else [c for c in d_candidates if c in df.columns]
    return FeatureConfig(pcols, fcols, dcols)

def build_multiinput_lstm_data(ohlcv_df: pd.DataFrame, value_df: pd.DataFrame, fund_df: pd.DataFrame, cfg: PipelineConfig):
    merged = merge_sources(ohlcv_df, value_df, fund_df)
    with_tgt = add_targets(merged, horizon=cfg.horizon, target_kind=cfg.target_kind)
    tr, va, te = time_split(with_tgt, cfg.split)
    feat = auto_feature_config(with_tgt, cfg)
    scalers = fit_scalers(tr, feat)
    tr_s = apply_scalers(tr, scalers, feat)
    va_s = apply_scalers(va, scalers, feat)
    te_s = apply_scalers(te, scalers, feat)
    Xp_tr, Xf_tr, Xd_tr, y_tr = build_windows(tr_s, feat, cfg.lookback)
    Xp_va, Xf_va, Xd_va, y_va = build_windows(va_s, feat, cfg.lookback)
    Xp_te, Xf_te, Xd_te, y_te = build_windows(te_s, feat, cfg.lookback)
    out = {
        "train": (Xp_tr, Xf_tr, Xd_tr, y_tr),
        "val":   (Xp_va, Xf_va, Xd_va, y_va),
        "test":  (Xp_te, Xf_te, Xd_te, y_te),
        "features": feat.__dict__,
        "scalers": {"price":"StandardScaler","flow":"StandardScaler","fund":"StandardScaler"},
    }
    # torch dataset(optional)
    if TORCH_AVAILABLE:
        out["torch"] = {
            "train": MultiInputTSDataset(Xp_tr, Xf_tr, Xd_tr, y_tr),
            "val":   MultiInputTSDataset(Xp_va, Xf_va, Xd_va, y_va),
            "test":  MultiInputTSDataset(Xp_te, Xf_te, Xd_te, y_te),
        }
    return out

# =============================================================
# 5) Example usage (replace with your real DataFrames)
# =============================================================
if __name__ == "__main__":
    print("This module provides functions to aggregate investor flows and build LSTM datasets.")
    print("Import and use build_multiinput_lstm_data(...) in your notebook/script.")