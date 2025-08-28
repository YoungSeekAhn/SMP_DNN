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
from get_global import get_global_stock_indices

from old.DSConfig import FeatureConfig

# # 마지막 거래일 기준으로 시작일과 종료일 설정
# # 시작일은 마지막 거래일로부터 2년 전으로 설정

def _standardize_date(df: pd.DataFrame,
                      date_cols: str = None,
                      out_name: str = "date") -> pd.DataFrame:
    """
    df에서 날짜 컬럼을 찾아 표준 컬럼명(out_name, 기본 'date')으로 반환.
    - 우선순위: (제공된 date_cols) -> ['date','data','Date','날짜','datetime','timestamp','time'] -> DatetimeIndex
    - 결과는 단일 컬럼 DataFrame (컬럼명 out_name)
    - timezone 제거(naive), 파싱 실패 행(NaT)은 제거
    """
    if df is None or df.empty:
        raise ValueError("Empty DataFrame passed to _standardize_date")

    # 1) 후보군 구성
    candidates = []
    if date_cols:
        candidates.extend(list(date_cols))
    candidates.extend(["date", "Date", "날짜", "datetime", "timestamp", "time"])

    # 2) 컬럼에서 찾기
    found_col = next((c for c in candidates if c in df.columns), None)

    if found_col is not None:
        s['date'] = df[found_col]
    else:
        # 3) 인덱스가 날짜이면 인덱스 사용
        if pd.api.types.is_datetime64_any_dtype(df.index):
            s = pd.Series(df.index, index=df.index, name=out_name)
        else:
            raise KeyError(
                "No date-like column found. "
                f"Tried: {candidates}. Index is not datetime."
            )

    # 6) 단일 컬럼 DataFrame으로 반환
    return s.to_frame()

def _standardize_ohlcv(ohlcv: pd.DataFrame, price_cols) -> pd.DataFrame:
    """
    OHLCV 컬럼 표준화:
      - 한국어/대소문자 다양한 표기를 영문 표준(open, high, low, close, volume, value, chg_pct)으로 통일
      - price_cols(영문)가 이미 있으면 그대로 사용
      - 숫자형 변환
    Datetime 인덱스 강제 없음.
    """
    if ohlcv is None or len(ohlcv) == 0:
        return ohlcv

    df = ohlcv.copy()

    # 한국어/영문 후보 매핑
    candidates = {
        "open":    ["시가", "Open", "open"],
        "high":    ["고가", "High", "high"],
        "low":     ["저가", "Low", "low"],
        "close":   ["종가", "Close", "close"],
        "volume":  ["거래량", "Volume", "volume"],
        "value":   ["거래대금", "Value", "value"],
        "chg_pct": ["등락률", "chg_pct", "Change%", "change_pct"],
    }

    name_map = {}
    for en, ko_list in candidates.items():
        for k in ko_list:
            if k in df.columns:
                name_map[k] = en
                break

    # price_cols에 이미 영문 키가 있으면 그대로 사용
    if price_cols:
        for en in price_cols:
            if en in df.columns and en not in name_map.values():
                # 이미 영문 컬럼명이면 그대로 두되, rename 대상엔 추가 안 함
                name_map.setdefault(en, en)

    # 선택 및 리네임
    select_cols = list(name_map.keys())
    if select_cols:
        df_sel = df[select_cols].rename(columns=name_map).copy()
    else:
        # 매핑이 하나도 안 되면 숫자형 컬럼만 유지
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        df_sel = df[num_cols].copy()

    # 숫자 변환
    for c in df_sel.columns:
        df_sel[c] = pd.to_numeric(df_sel[c], errors="coerce")

    return df_sel

def _standardize_investor(value_df: pd.DataFrame, flow_cols) -> pd.DataFrame:
    """
    투자주체 컬럼 표준화:
      - 한국어 컬럼명을 영어로 매핑(기관합계/개인/외국인/기타법인 등)
      - 주어진 flow_cols가 이미 영어로 들어온 경우 그대로 사용
      - 숫자형으로 변환
    Datetime 인덱스 강제 없음.
    """
    if value_df is None or len(value_df) == 0:
        return value_df

    df = value_df.copy()

    # 한국어 -> 영문 후보 매핑
    candidates = {
        "inst_sum": ["기관합계", "기관", "기관투자자"],
        "inst_ext": ["기타법인", "법인", "법인투자자"],
        "retail":   ["개인", "개인합계", "개인투자자"],
        "foreign":  ["외국인", "외국인합계", "외국인투자자"],
    }

    name_map = {}
    for en, ko_list in candidates.items():
        for k in ko_list:
            if k in df.columns:
                name_map[k] = en
                break

    # flow_cols에 이미 영어 키가 있으면 그대로 사용
    if flow_cols:
        for en in flow_cols:
            if en in df.columns and en not in name_map.values():
                # 이미 영문 컬럼명이면 그대로 두되, rename 대상엔 추가 안 함
                name_map.setdefault(en, en)

    # 선택 및 리네임
    select_cols = list(name_map.keys())
    if select_cols:
        df_sel = df[select_cols].rename(columns=name_map).copy()
    else:
        # 매핑이 하나도 안 되면 숫자형 컬럼만 유지
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        df_sel = df[num_cols].copy()

    # 숫자 변환
    for c in df_sel.columns:
        df_sel[c] = pd.to_numeric(df_sel[c], errors="coerce")

    return df_sel

def _standardize_fundamental(fund_df: pd.DataFrame, fund_cols) -> pd.DataFrame:
    """
    펀더멘털 컬럼 표준화:
      - 한국어/대소문자 다양한 표기를 표준 소문자(bps, per, pbr, eps, div, dps)로 통일
      - fund_cols가 주어지면 그 교집합만 선택
      - 숫자형으로 변환
    Datetime 인덱스 강제 없음.
    """
    if fund_df is None or len(fund_df) == 0:
        return fund_df

    df = fund_df.copy()

    rename_map = {
        "BPS": "bps", "PER": "per", "PBR": "pbr", "EPS": "eps", "DIV": "div", "DPS": "dps",
        "bps": "bps", "per": "per", "pbr": "pbr", "eps": "eps", "div": "div", "dps": "dps",
        "Bps": "bps", "Per": "per", "Pbr": "pbr", "Eps": "eps", "Div": "div", "Dps": "dps",
    }
    df = df.rename(columns=rename_map)

    # 유지할 컬럼 선택
    if fund_cols:
        keep = [c for c in fund_cols if c in df.columns]
        if keep:
            df = df[keep].copy()
        else:
            # fund_cols가 비거나 교집합이 없으면 표준 키들 중 존재하는 것만
            std = ["per", "pbr", "div", "bps", "eps", "dps"]
            keep2 = [c for c in std if c in df.columns]
            df = df[keep2].copy() if keep2 else df.copy()
    else:
        # fund_cols가 없으면 표준 키 우선
        std = ["per", "pbr", "div", "bps", "eps", "dps"]
        keep = [c for c in std if c in df.columns]
        if keep:
            df = df[keep].copy()

    # 숫자 변환
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df

def merge_sources(date: pd.DataFrame, ohlcv: pd.DataFrame, investor: pd.DataFrame, fund: pd.DataFrame) -> pd.DataFrame:
    """Outer-join on date, then sort."""
    dfs = []
    if date is not None and len(date):
        dfs.append(date)
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


def get_dataset(cfg, SAVE_CSV_FILE=True, PLOT_ROLLING=True) -> pd.DataFrame:
    """
    Fetches and merges stock data from KRX, including OHLCV, investor trading values, and fundamentals.
    
    Args:
        cfg (config): Configuration object with dataset parameters.
        feature (FeatureConfig): Feature configuration for column names.
    
    Returns:
        pd.DataFrame: Merged DataFrame with standardized columns.
    """
    feature = FeatureConfig()

    Start_Date = cfg.start_date
    End_Date = cfg.end_date
    Code = cfg.code  # 종목 코드

    print(f"Stock Name: {cfg.name}, Code: {Code} /n")
    print(f"Start Day: {Start_Date}, End Day: {End_Date}")
    
     # # 1) Fetch OHLCV data
    OHLCV_df = stock.get_market_ohlcv(Start_Date, End_Date, Code)
    OHLCV_df = _standardize_ohlcv(OHLCV_df, feature.price_cols)
    
    # # 2) Fetch investor trading values
    Investor_df = stock.get_market_trading_value_by_date(Start_Date, End_Date, Code)
    Investor_df = _standardize_investor(Investor_df, feature.flow_cols)

    # # 3) Fetch fundamental data
    Fund_df = stock.get_market_fundamental(Start_Date, End_Date, Code)
    Fund_df = _standardize_fundamental(Fund_df, feature.fund_cols)
    
    Date_df = _standardize_date(OHLCV_df, feature.date_cols)
    # # 4) Merge all sources
    merge_stock_df = merge_sources(Date_df, OHLCV_df, Investor_df, Fund_df)
    
    global_df = get_global_stock_indices(cfg)
    merged_df = pd.merge(merge_stock_df, global_df, on="date", how="left")
    

    
    if not merged_df.empty and float(merged_df['inst_sum'].iloc[-1]) == 0:
        print("Today trading data not validated. Removing today data row.")
        merged_df = merged_df.iloc[:-1].copy()
        
    
    print(f"Data shape after merge: {merged_df.shape}")
    print(f"Columns after merge: {merged_df.columns.tolist()}")
    
    # 8) (선택) rolling h=1 경로
    if SAVE_CSV_FILE:
        print(merged_df.head())
        get_dir = Path(cfg.getdata_dir)
        get_dir.mkdir(exist_ok=True, parents=True)
        filepath = os.path.join(get_dir, f"{cfg.name}({cfg.code})_{cfg.end_date}.csv")
        merged_df.to_csv(filepath, index=True)

    if PLOT_ROLLING:
        # Plot rolling averages and other features
        if merged_df.empty:
            raise ValueError("Merged DataFrame is empty. Please check the data sources.")
        
        plot_rolling_features(merged_df, cfg, feature)
    
    return merged_df

def plot_rolling_features(merge_df: pd.DataFrame, cfg, feature: FeatureConfig):
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
    Investor_colums = [col for col in merge_df.columns if col in feature.flow_cols]
    for col in Investor_colums:
        if col in Investor_colums:
            axes[1].plot(
                merge_df.index, 
                merge_df[col].rolling(20).mean(), 
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
