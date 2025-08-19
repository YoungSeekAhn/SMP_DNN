import os
os.environ.pop("MPLBACKEND", None)  # 환경변수 꼬임 방지 (옵션)

import matplotlib
matplotlib.use("TkAgg")   # 또는 "QtAgg" (PyQt5/PySide6 설치 필요)
import matplotlib.pyplot as plt

from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from pathlib import Path

import pandas as pd
import numpy as np
from pykrx import stock

from DSConfig import DSConfig,FeatureConfig

SAVE_CSV_FILE = True  # CSV 파일 저장 여부`
PLOT_ROLLING = True  # 롤링 차트 플로팅 여부

# 마지막 거래일 기준으로 시작일과 종료일 설정
# 시작일은 마지막 거래일로부터 2년 전으로 설정
cfg = DSConfig()
feature = FeatureConfig()

Start_Date = cfg.start_date
End_Date = cfg.end_date
Code = cfg.code  # 종목 코드

print(f"Stock Name: {cfg.name}, Code: {Code} /n")
print(f"Start Day: {Start_Date}, End Day: {End_Date}")


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

def _standardize_ohlcv(ohlcv: pd.DataFrame,price_cols) -> pd.DataFrame:
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

def _standardize_investor(value_df: pd.DataFrame, flow_cols) -> pd.DataFrame:
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

def _standardize_fundamental(fund_df: pd.DataFrame, fund_cols) -> pd.DataFrame:
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

# KRX 주식 데이터 가져오기  

# Example: Get trading value by date for Samsung Electronics (005930)


# Get Open, High, Low, Close, Volumn data by date
OHLCV_df = stock.get_market_ohlcv(Start_Date, End_Date, Code)
OHLCV_df = _standardize_ohlcv(OHLCV_df, feature.price_cols)
# Get trading value by date
#Investor_df = stock.get_market_trading_value_by_date(Start_Date, End_Date, Code, etf=True, etn=True, elw=True, detail=True)
Investor_df = stock.get_market_trading_value_by_date(Start_Date, End_Date, Code)
Investor_df = _standardize_investor(Investor_df, feature.flow_cols)
# fundamental data by date (e.g., P/E, P/B, Dividend Yield)
Fund_df = stock.get_market_fundamental(Start_Date, End_Date, Code)
Fund_df = _standardize_fundamental(Fund_df, feature.fund_cols)

merge_df = merge_sources(OHLCV_df, Investor_df, Fund_df)


# 8) (선택) rolling h=1 경로
if SAVE_CSV_FILE:
    print(merge_df.head())
    get_dir = Path(cfg.getdata_dir)
    get_dir.mkdir(exist_ok=True, parents=True)
    filepath = os.path.join(get_dir, f"{cfg.name}({cfg.code})_{cfg.end_date}.csv")
    merge_df.to_csv(filepath, index=True)

if PLOT_ROLLING:
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # (1) 주가
    close = merge_df['close']
    if close.empty:
        raise ValueError("Close price data is empty. Please check the data source.")
    axes[0].plot(close.index, close, label="close", color="black")
    axes[0].plot(close.rolling(1).mean(), label="MA1", linestyle="--")
    axes[0].plot(close.rolling(5).mean(), label="MA5", linestyle="--")
    axes[0].plot(close.rolling(20).mean(), label="MA20", linestyle="--")
    axes[0].plot(close.rolling(60).mean(), label="MA60", linestyle="--")

    axes[0].set_title(f"{cfg.name} ({cfg.code}) 종가 with Moving Averages")
    axes[0].set_ylabel("Price")
    axes[0].legend()
    axes[0].grid(True)

    # (2) 수급주체

    #수급주체별 20일 이동평균을 플로팅
    #investor_cols : merge_df 안에서 수급주체별 컬럼명 리스트

    for col in Investor_df.columns:
        if col in Investor_df.columns:
            axes[1].plot(
                Investor_df.index, 
                Investor_df[col].rolling(20).mean(), 
                label=f"{col} 20MA"
            )

    axes[1].set_title(f"{cfg.name} ({cfg.code}) Investor Trading Value (20-day Moving Average)")
    axes[1].set_ylabel("Trading Value")
    axes[1].legend()
    axes[1].grid(True)


    # (3) PER / PBR
    per = merge_df['per']
    pbr = merge_df['pbr']
    if per.empty or pbr.empty:
        raise ValueError("PER or PBR data is empty. Please check the data source.")
    axes[2].plot(per.rolling(20).mean(), label="PER MA20", color="blue")
    axes[2].plot(pbr.rolling(20).mean()*10, label="PBR MA20 (Scale *10)", color="red")
    axes[2].set_title(f"{cfg.name} ({cfg.code}) PER / PBR 추이")
    axes[2].set_ylabel("값")
    axes[2].legend()
    axes[2].grid(True)

    # 전체 스타일
    plt.xlabel("날짜")
    plt.tight_layout()
    plt.show()
