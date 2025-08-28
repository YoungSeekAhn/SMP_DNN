
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime, timedelta
from pykrx import stock
from old.DSConfig import config
from dataset_functions import last_trading_day
from old.get_dataset import get_dataset
from old.make_dataset import make_datasets
from train_lstm import training_LSTM

merged_df = None  # Placeholder for the merged DataFrame
payload = None  # Placeholder for the dataset payload
LOAD_CSV_FILE = True  # Load from CSV if available
SAVE_CSV_FILE = False  # Do not save CSV in this run
PLOT_ROLLING = False  # Do not plot rolling charts in this run

Stock_Name = "현대차"  # 종목 이름
Stock_Code = ""  # 종목 코드 (자동설정)
#Stock_Code = "005930"  # 종목 코드 (자동설정)
# 전체 종목 리스트 (코드와 이름)
if not Stock_Code:
    tickers = stock.get_market_ticker_list(market="ALL")  # KOSPI, KOSDAQ 모두
    mapping = {stock.get_market_ticker_name(t): t for t in tickers}
    Stock_Code = mapping[Stock_Name]
if not Stock_Code:
    raise ValueError(f"종목 이름 '{Stock_Name}'에 해당하는 종목 코드가 없습니다. 종목 이름을 확인하세요.")

# 시작일과 종료일 자동 설정   
Duration = 365 * 2  # 데이터 기간 (일수)     
End_Date = last_trading_day()
Start_Date = (datetime.strptime(End_Date, "%Y%m%d") - timedelta(days=Duration)).strftime("%Y%m%d")

config.name = Stock_Name       # 종목 이름
config.code = Stock_Code   # 종목 코드 (자동설정)   
config.duration = Duration  # 데이터 기간 (일수)
config.start_date = Start_Date  # 자동 결정 (예: "20220101")    
config.end_date = End_Date    # 자동 결정 (예: "20231231")

get_dir = Path(config.getdata_dir)
csvpath = os.path.join(get_dir, f"{config.name}({config.code})_{config.end_date}.csv")

if not os.path.exists(csvpath):   # 파일이 없으면
    merged_df = get_dataset(config, SAVE_CSV_FILE=True)
    print("CSV 수집 데이터 파일을 생성합니다.:", csvpath)

else:
    print("CSV 수집 데이터 파일이 이미 존재 합니다:", csvpath)

dataset_dir = Path(config.dataset_dir)
datapath = os.path.join(dataset_dir, f"{config.name}({config.code})_{config.end_date}.pkl")

if not os.path.exists(datapath):   # 파일이 없으면
    payload = make_datasets(merged_df, config, LOAD_CSV_FILE=True)
    print("학습 데이터셋 파일을 생성합니다.:", datapath)

else:
    print("학습 데이터셋 파일이 이미 존재 합니다:", datapath)
    
training_LSTM(payload, config)
