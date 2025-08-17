import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pykrx import stock

# -----------------------------
# 1) Standardize & Merge
# -----------------------------
        
def _ensure_datetime_index(df: pd.DataFrame, date_col_candidates=("날짜","date","Date")) -> pd.DataFrame:
    """Make sure dataframe has a DatetimeIndex named 'date'."""
    if df is None or len(df)==0:
        return df
    dfc = df.copy()
    if isinstance(dfc.index, pd.DatetimeIndex):
        dfc.index.name = "date"
        return dfc.sort_index()
    for c in date_col_candidates:
        if c in dfc.columns:
            dfc[c] = pd.to_datetime(dfc[c])
            dfc = dfc.set_index(c)
            dfc.index.name = "date"
            return dfc.sort_index()
    # last resort: try to parse current index
    try:
        idx = pd.to_datetime(dfc.index)
        dfc.index = idx
        dfc.index.name = "date"
        return dfc.sort_index()
    except Exception:
        raise ValueError("No datetime index or date column found. Provide a date column named one of: %s" % (date_col_candidates,))

def _standardize_ohlcv(ohlcv: pd.DataFrame) -> pd.DataFrame:
    if ohlcv is None or len(ohlcv)==0:
        return ohlcv
    df = _ensure_datetime_index(ohlcv)
    rename_map = {
        "시가":"open", "고가":"high", "저가":"low", "종가":"close",
        "거래량":"volume", "거래대금":"value", "등락률":"chg_pct",
        "Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume","Value":"value",
        "open":"open","high":"high","low":"low","close":"close","volume":"volume","value":"value"
    }
    df = df.rename(columns=rename_map)
    # keep common columns if exist
    keep = [c for c in ["open","high","low","close","volume","value","chg_pct"] if c in df.columns]
    df = df[keep]
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _standardize_investor(value_df: pd.DataFrame) -> pd.DataFrame:
    """Select key investor groups and rename to english. Falls back if some missing."""
    if value_df is None or len(value_df)==0:
        return value_df
    df = _ensure_datetime_index(value_df)
    # Common column names in PyKRX (detail=True adds more columns)
    # We'll try to pick ["개인", "외국인", "기관합계"] or reasonable fallbacks.
    name_map = {}
    candidates = {
        "inst_sum": ["기관합계","기관","기관투자자"],
        "inst_ext": ["기타법인","법인","법인투자자"],
        "retail": ["개인","개인합계","개인투자자"],
        "foreign": ["외국인","외국인합계","외국인투자자"],
    }
    for en, ko_list in candidates.items():
        for k in ko_list:
            if k in df.columns:
                name_map[k] = en
                break
    # If none found, try lowercase english direct
    for en in ["inst_sum","inst_ext","retail","foreign"]:
        if en not in name_map.values() and en in df.columns:
            name_map[en] = en
    # subset/rename
    select_cols = list(name_map.keys())
    if not select_cols:
        # keep all numeric columns but warn
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        df_sel = df[num_cols].copy()
    else:
        df_sel = df[select_cols].rename(columns=name_map).copy()
    # numeric
    for c in df_sel.columns:
        df_sel[c] = pd.to_numeric(df_sel[c], errors="coerce")
    return df_sel

def _standardize_fundamental(fund_df: pd.DataFrame) -> pd.DataFrame:
    if fund_df is None or len(fund_df)==0:
        return fund_df
    df = _ensure_datetime_index(fund_df)
    rename_map = {
        "BPS":"bps","PER":"per","PBR":"pbr","EPS":"eps","DIV":"div","DPS":"dps",
        "bps":"bps","per":"per","pbr":"pbr","eps":"eps","div":"div","dps":"dps"
    }
    df = df.rename(columns=rename_map)
    keep = [c for c in ["per","pbr","div","bps","eps","dps"] if c in df.columns]
    df = df[keep]
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # fundamentals are often sparse (monthly/quarterly) -> forward-fill
    df = df.sort_index().ffill()
    return df

def merge_sources(ohlcv: pd.DataFrame, investor: pd.DataFrame, fund: pd.DataFrame) -> pd.DataFrame:
    """Outer-join on date, then sort."""
    dfs = []
    if ohlcv is not None and len(ohlcv):
        #dfs.append(_standardize_ohlcv(ohlcv))
        dfs.append(ohlcv)
    if investor is not None and len(investor):
        #dfs.append(_standardize_investor(investor))
        dfs.append(investor)
    if fund is not None and len(fund):
        #dfs.append(_standardize_fundamental(fund))
        dfs.append(fund)
    if not dfs:
        raise ValueError("No input dataframes provided.")
    out = dfs[0]
    for d in dfs[1:]:
        out = out.join(d, how="outer")
    out = out.sort_index()
    # basic cleaning
    # drop days without close; forward-fill fundamentals already handled
    if "close" in out.columns:
        out = out[out["close"].notna()]
    # optional: fill investor NaN with 0 (no trade recorded)
    for c in ["inst_sum","inst_ext","retail","foreign"]:
        if c in out.columns:
            out[c] = out[c].fillna(0.0)
    # ensure no inf
    out = out.replace([np.inf,-np.inf], np.nan)
    out = out.dropna(subset=["close"])
    return out

# -----------------------------
# 2) Feature/Target & Splits
# -----------------------------

@dataclass
class SplitConfig:
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

# def add_targets(df: pd.DataFrame, horizon: int = 1, target_kind: str = "logr") -> pd.DataFrame:
#     """
#     target_kind:
#       - 'logr' : log return log(C_{t+H}/C_t)
#       - 'close': raw future close C_{t+H}
#       - 'direction': 1 if future log return > 0 else 0
#     """
#     out = df.copy()
#     if "close" not in out.columns:
#         raise ValueError("close column required to compute targets.")
#     future_close = out["close"].shift(-horizon)
#     if target_kind == "logr":
#         out["target"] = np.log(future_close / out["close"])
#     elif target_kind == "close":
#         out["target"] = future_close
#     elif target_kind == "direction":
#         out["target"] = (np.log(future_close / out["close"]) > 0).astype(float)
#     else:
#         raise ValueError("Unsupported target_kind")
#     out = out.dropna(subset=["target"])
#     return out

def add_multi_targets(df: pd.DataFrame, horizons: list = [1], target_kind: str = "logr") -> pd.DataFrame:
    """
    horizons: 예측할 미래 시점 리스트 (예: [1,2,3,4,5])
    target_kind:
      - 'logr' : log return log(C_{t+H}/C_t)
      - 'close': raw future close C_{t+H}
      - 'direction': 1 if future log return > 0 else 0
    """
    out = df.copy()
    if "close" not in out.columns:
        raise ValueError("close column required to compute targets.")
    for h in horizons:
        future_close = out["close"].shift(-h)
        if target_kind == "logr":
            out[f"target_h{h}"] = np.log(future_close / out["close"])
        elif target_kind == "close":
            out[f"target_h{h}"] = future_close
        elif target_kind == "direction":
            out[f"target_h{h}"] = (np.log(future_close / out["close"]) > 0).astype(float)
        else:
            raise ValueError("Unsupported target_kind")
    # drop rows where any target is NaN
    target_cols = [f"target_h{h}" for h in horizons]
    out = out.dropna(subset=target_cols)
    return out


def time_split(df: pd.DataFrame, split: SplitConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    i_tr = int(n * split.train_ratio)
    i_va = int(n * (split.train_ratio + split.val_ratio))
    tr = df.iloc[:i_tr].copy()
    va = df.iloc[i_tr:i_va].copy()
    te = df.iloc[i_va:].copy()
    return tr, va, te

# -----------------------------
# 3) Scaling & Windows
# -----------------------------

@dataclass
class FeatureConfig:
    price_cols: List[str]
    flow_cols: List[str]
    fund_cols: List[str]

class BranchScalers:
    def __init__(self):
        self.price = StandardScaler()
        self.flow  = StandardScaler()
        self.fund  = StandardScaler()
    def state_dict(self) -> Dict:
        return {"price": self.price, "flow": self.flow, "fund": self.fund}

def fit_scalers(train: pd.DataFrame, feat: FeatureConfig) -> BranchScalers:
    sc = BranchScalers()
    if feat.price_cols:
        sc.price.fit(train[feat.price_cols].values)
    if feat.flow_cols:
        sc.flow.fit(train[feat.flow_cols].values)
    if feat.fund_cols:
        sc.fund.fit(train[feat.fund_cols].values)
    return sc

def apply_scalers(df: pd.DataFrame, sc: BranchScalers, feat: FeatureConfig) -> pd.DataFrame:
    out = df.copy()
    if feat.price_cols:
        out.loc[:, feat.price_cols] = sc.price.transform(out[feat.price_cols].values)
    if feat.flow_cols:
        out.loc[:, feat.flow_cols] = sc.flow.transform(out[feat.flow_cols].values)
    if feat.fund_cols:
        out.loc[:, feat.fund_cols] = sc.fund.transform(out[feat.fund_cols].values)
    return out

def build_windows(df: pd.DataFrame, feat: FeatureConfig, lookback, horizons) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    Xp, Xf, Xd, y = [], [], [], []
    values_price = df[feat.price_cols].values if feat.price_cols else None
    values_flow  = df[feat.flow_cols].values  if feat.flow_cols  else None
    values_fund  = df[feat.fund_cols].values  if feat.fund_cols  else None
    
    target_cols = [f"target_h{h}" for h in horizons]
    targets = df[target_cols].values  # shape: [N, num_horizons]
    
    L = lookback
    N = len(df)
    # windows end at t; predict t+H (already embedded in target)
    for t in range(L-1, N):
        # the target at index t corresponds to horizon applied already; but last H rows were dropped
        xp = values_price[t-L+1:t+1] if values_price is not None else None
        xf = values_flow[t-L+1:t+1]  if values_flow is not None else None
        xd = values_fund[t-L+1:t+1]  if values_fund is not None else None
        yt = targets[t]
        # skip windows that include NaN
        ok = True
        for arr in [xp, xf, xd]:
            if arr is not None and (np.isnan(arr).any() or len(arr) < L):
                ok = False
                break
            
        # also skip if target has NaN
        if ok and not np.isnan(yt).any():
            if xp is not None: Xp.append(xp)
            else: Xp.append(np.zeros((L,0), dtype=np.float32))
            if xf is not None: Xf.append(xf)
            else: Xf.append(np.zeros((L,0), dtype=np.float32))
            if xd is not None: Xd.append(xd)
            else: Xd.append(np.zeros((L,0), dtype=np.float32))
            y.append(yt)
            
    Xp = np.asarray(Xp, dtype=np.float32)
    Xf = np.asarray(Xf, dtype=np.float32)
    Xd = np.asarray(Xd, dtype=np.float32)
    y  = np.asarray(y,  dtype=np.float32)
    return Xp, Xf, Xd, y

# -----------------------------
# 4) PyTorch Dataset
# -----------------------------

class MultiInputTSDataset(torch.utils.data.Dataset):
    def __init__(self, Xp, Xf, Xd, y, lookback, horizons):
        """
        Xp, Xf, Xd: np.ndarray (N, F)
        y:  (N,), (N,1)  또는 (N, H)  # H = len(horizons)인 다중 horizon 타깃도 허용
        lookback: int
        horizons: int | list[int]     # 예: 1 또는 [1,2,3]
        """
        self.lookback = int(lookback)

        # horizons 표준화
        if isinstance(horizons, (list, tuple, np.ndarray)):
            self.horizons = [int(h) for h in horizons]
        else:
            self.horizons = [int(horizons)]
        H = len(self.horizons)
        horizon_max = max(self.horizons)

        # 배열화
        Xp = np.asarray(Xp); Xf = np.asarray(Xf); Xd = np.asarray(Xd)
        y  = np.asarray(y)

        # y 형태 판별
        if y.ndim == 1:                            # (N,)
            mode = "single_series"
            needed_future = horizon_max
        elif y.ndim == 2 and y.shape[1] == 1:      # (N,1) -> (N,)
            y = y[:, 0]
            mode = "single_series"
            needed_future = horizon_max
        elif y.ndim == 2 and y.shape[1] == H:      # (N, H) 이미 다중 horizon 정렬됨
            mode = "multi_ready"
            needed_future = 1                      # 한 행만 쓰면 됨
        else:
            raise ValueError(
                f"y shape 불일치: 기대 (N,), (N,1), (N,{H}) 중 하나, 현재 {y.shape}"
            )

        # 사용 가능한 시작 인덱스
        max_start = len(y) - self.lookback - needed_future + 1
        if max_start <= 0:
            raise ValueError(
                f"데이터 길이 부족: len(y)={len(y)}, lookback={self.lookback}, "
                f"needed_future={needed_future}"
            )

        # 윈도우 슬라이싱
        Xp_list, Xf_list, Xd_list, y_list = [], [], [], []
        for i in range(max_start):
            Xp_list.append(Xp[i:i+self.lookback])
            Xf_list.append(Xf[i:i+self.lookback])
            Xd_list.append(Xd[i:i+self.lookback])

            if mode == "single_series":
                # 원시 단일 시계열에서 horizons 만큼 미래 타깃 직접 수집
                targets = [y[i+self.lookback+h-1] for h in self.horizons]  # 길이 H
            else:
                # (N,H) 이미 정렬된 경우: 현재 앵커의 한 행을 그대로 사용
                targets = y[i+self.lookback-1, :]  # shape (H,)
            y_list.append(targets)

        # 배열화 & shape 고정: (N_samples, L, F), y: (N_samples, H)
        self.Xp = np.array(Xp_list).reshape(-1, self.lookback, Xp.shape[-1])
        self.Xf = np.array(Xf_list).reshape(-1, self.lookback, Xf.shape[-1])
        self.Xd = np.array(Xd_list).reshape(-1, self.lookback, Xd.shape[-1])
        self.y  = np.array(y_list, dtype=np.float32).reshape(-1, H)

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return {
            "x_price": torch.tensor(self.Xp[idx], dtype=torch.float32),
            "x_flow":  torch.tensor(self.Xf[idx], dtype=torch.float32),
            "x_fund":  torch.tensor(self.Xd[idx], dtype=torch.float32),
            "y":       torch.tensor(self.y[idx],  dtype=torch.float32),  # (H,)
        }

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return {
            "x_price": torch.tensor(self.Xp[idx], dtype=torch.float32),
            "x_flow":  torch.tensor(self.Xf[idx], dtype=torch.float32),
            "x_fund":  torch.tensor(self.Xd[idx], dtype=torch.float32),
            "y":       torch.tensor(self.y[idx],  dtype=torch.float32),  # (len(horizons),)
        }
@dataclass
class PipelineConfig:
    lookback: int = 20
    horizons: List[int] = field(default_factory=lambda: [1])  # 다중 타깃 지원
    target_kind: str = "logr"
    split: SplitConfig = field(default_factory=SplitConfig)    # <- default_factory 사용
    price_cols: Optional[List[str]] = None  # None -> auto
    flow_cols: Optional[List[str]]  = None  # None -> auto
    fund_cols: Optional[List[str]]  = None  # None -> auto
    

def auto_feature_config(df: pd.DataFrame, override: PipelineConfig) -> FeatureConfig:
    price_candidates = ["open","high","low","close","volume","value","chg_pct"]
    flow_candidates  = ["inst_sum","inst_ext","retail","foreign"]
    fund_candidates  = ["per","pbr","div","bps","eps","dps"]
    pcols = override.price_cols if override.price_cols is not None else [c for c in price_candidates if c in df.columns]
    fcols = override.flow_cols  if override.flow_cols  is not None else [c for c in flow_candidates  if c in df.columns]
    dcols = override.fund_cols  if override.fund_cols  is not None else [c for c in fund_candidates  if c in df.columns]
    return FeatureConfig(price_cols=pcols, flow_cols=fcols, fund_cols=dcols)

