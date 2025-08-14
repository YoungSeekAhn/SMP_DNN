import os
os.environ.pop("MPLBACKEND", None)  # 환경변수 꼬임 방지 (옵션)

import matplotlib
matplotlib.use("TkAgg")   # 또는 "QtAgg" (PyQt5/PySide6 설치 필요)
import matplotlib.pyplot as plt

from datetime import date
import pandas as pd
from pykrx import stock

from dataset_functions import merge_sources, _standardize_ohlcv, _standardize_fundamental, _standardize_investor

Start_Date = "20240801"
End_Date = "20250812"

# Example: Get trading value by date for Samsung Electronics (005930)
Code = "005930"

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

from pathlib import Path

print(merge_df.head())
out_dir = Path.cwd()
filepath = os.path.join(out_dir, f"{Code}_merged_data.csv")
merge_df.to_csv(filepath, index=True)
