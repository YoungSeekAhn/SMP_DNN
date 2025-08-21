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
def is_trading_day(yyyymmdd: str, ticker: str = "005930") -> bool:
    """해당 날짜가 거래일인지 여부 반환 (티커 일봉 데이터 존재 여부로 판단)."""
    df = stock.get_market_ohlcv_by_date(yyyymmdd, yyyymmdd, ticker)
    return df is not None and len(df) > 0

def last_trading_day(ref: datetime | None = None) -> str:
    """
    기준일(ref) 포함하여, 가장 최근 거래일 'YYYYMMDD' 반환.
    ref가 None이면 오늘 기준.
    """
    if ref is None:
        ref = datetime.today()
    d = ref
    while True:
        ymd = d.strftime("%Y%m%d")
        if is_trading_day(ymd):
            return ymd
        d -= timedelta(days=1)
        
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

def _standardize_ohlcv(ohlcv: pd.DataFrame,price_cols: List[str]) -> pd.DataFrame:
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
    keep = [c for c in price_cols if c in df.columns]
    df = df[keep]
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _standardize_investor(value_df: pd.DataFrame, flow_cols: List[str]) -> pd.DataFrame:
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
    for en in flow_cols:
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

def _standardize_fundamental(fund_df: pd.DataFrame, fund_cols: List[str]) -> pd.DataFrame:
    if fund_df is None or len(fund_df)==0:
        return fund_df
    df = _ensure_datetime_index(fund_df)
    rename_map = {
        "BPS":"bps","PER":"per","PBR":"pbr","EPS":"eps","DIV":"div","DPS":"dps",
        "bps":"bps","per":"per","pbr":"pbr","eps":"eps","div":"div","dps":"dps"
    }
    df = df.rename(columns=rename_map)
    #keep = [c for c in ["per","pbr","div","bps","eps","dps"] if c in df.columns]
    keep = [c for c in fund_cols if c in df.columns]
    df = df[keep]
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # fundamentals are often sparse (monthly/quarterly) -> forward-fill
    df = df.sort_index().ffill()
    return df

def merge_sources(data: pd.DataFrame, ohlcv: pd.DataFrame, investor: pd.DataFrame, fund: pd.DataFrame) -> pd.DataFrame:
    """Outer-join on date, then sort."""
    dfs = []
    if data is not None and len(data):
        dfs.append(data)
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






