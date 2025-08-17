import os
os.environ.pop("MPLBACKEND", None)  # 환경변수 꼬임 방지 (옵션)

import matplotlib
matplotlib.use("TkAgg")   # 또는 "QtAgg" (PyQt5/PySide6 설치 필요)
import matplotlib.pyplot as plt

from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from pathlib import Path
from dataset_functions import _standardize_ohlcv, _standardize_investor, _standardize_fundamental
from dataset_functions import merge_sources

import pandas as pd
import numpy as np
from pykrx import stock

from DSConfig import DSConfig

SAVE_CSV_FILE = True  # CSV 파일 저장 여부`

# 마지막 거래일 기준으로 시작일과 종료일 설정
# 시작일은 마지막 거래일로부터 2년 전으로 설정
cfg = DSConfig(
    code="005930",
    lookback=30,
    horizons=[1, 2, 5],
    target_kind="logr",  # "logr" | "pct" | "close"
    start_date=None,   # 자동 결정
    end_date=None,     # 자동 결정
)

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

Duration = 365 * 2  # 2년
# 시작일과 종료일 자동 설정        
End_Date = last_trading_day()
Start_Date = (datetime.strptime(End_Date, "%Y%m%d") - timedelta(days=Duration)).strftime("%Y%m%d")
print(f"End Day: {End_Date}, Start Day: {Start_Date}")


# 1) KRX 주식 데이터 가져오기

# KRX 주식 데이터 가져오기  

# Example: Get trading value by date for Samsung Electronics (005930)
Code = cfg.code

# Get Open, High, Low, Close, Volumn data by date
OHLCV_df = stock.get_market_ohlcv(Start_Date, End_Date, Code)
OHLCV_df = _standardize_ohlcv(OHLCV_df)
# Get trading value by date
#Investor_df = stock.get_market_trading_value_by_date(Start_Date, End_Date, Code, etf=True, etn=True, elw=True, detail=True)
Investor_df = stock.get_market_trading_value_by_date(Start_Date, End_Date, Code)
Investor_df = _standardize_investor(Investor_df)
# fundamental data by date (e.g., P/E, P/B, Dividend Yield)
Fund_df = stock.get_market_fundamental(Start_Date, End_Date, Code)
Fund_df = _standardize_fundamental(Fund_df)

merge_df = merge_sources(OHLCV_df, Investor_df, Fund_df)


# 8) (선택) rolling h=1 경로
if SAVE_CSV_FILE:
    print(merge_df.head())
    out_dir = Path.cwd()
    filepath = os.path.join(out_dir, f"{Code}_merged_data.csv")
    merge_df.to_csv(filepath, index=True)

def plot_close_with_ma(merge_df, close_col="close"):
    """
    종가와 1일, 5일, 20일, 60일 이동평균선을 플로팅
    close_col : merge_df 안에서 종가 컬럼명 ("Close" 또는 "종가" 등)
    """
    if close_col not in merge_df.columns:
        raise ValueError(f"{close_col} 컬럼이 merge_df에 없습니다. merge_df.columns 확인 필요")

    close = merge_df[close_col]

    plt.figure(figsize=(14,6))
    plt.plot(close.index, close, label="close", color="black")

    plt.plot(close.rolling(1).mean(), label="MA1", linestyle="--")
    plt.plot(close.rolling(5).mean(), label="MA5", linestyle="--")
    plt.plot(close.rolling(20).mean(), label="MA20", linestyle="--")
    plt.plot(close.rolling(60).mean(), label="MA60", linestyle="--")

    plt.title("Close Price with Moving Averages")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_investor_ma20(Investor_df, investor_cols):
    """
    수급주체별 20일 이동평균을 플로팅
    investor_cols : merge_df 안에서 수급주체별 컬럼명 리스트
    """
    plt.figure(figsize=(14,6))
    
    for col in investor_cols:
        if col in Investor_df.columns:
            plt.plot(
                Investor_df.index, 
                Investor_df[col].rolling(20).mean(), 
                label=f"{col} 20MA"
            )

    plt.title("Investor Trading Value (20-day Moving Average)")
    plt.xlabel("Date")
    plt.ylabel("Trading Value")
    plt.legend()
    plt.grid(True)
    plt.show()
    
# 사용 예시
plot_close_with_ma(merge_df, close_col="close")   # 또는 close_col="종가"
plot_investor_ma20(Investor_df, Investor_df.columns)
 