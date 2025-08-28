# DSConfig.py
from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime, timedelta
from pykrx import stock


# Stock_Name = "LG전자"  # 종목 이름

# # 전체 종목 리스트 (코드와 이름)
# tickers = stock.get_market_ticker_list(market="ALL")  # KOSPI, KOSDAQ 모두
# mapping = {stock.get_market_ticker_name(t): t for t in tickers}

# Stock_Code = mapping[Stock_Name]

# if not Stock_Code:
#     raise ValueError(f"종목 이름 '{Stock_Name}'에 해당하는 종목 코드가 없습니다. 종목 이름을 확인하세요.")

# def is_trading_day(yyyymmdd: str, ticker: str = "005930") -> bool:
#     """해당 날짜가 거래일인지 여부 반환 (티커 일봉 데이터 존재 여부로 판단)."""
#     df = stock.get_market_ohlcv_by_date(yyyymmdd, yyyymmdd, ticker)
#     return df is not None and len(df) > 0

# def last_trading_day(ref: datetime | None = None) -> str:
#     """
#     기준일(ref) 포함하여, 가장 최근 거래일 'YYYYMMDD' 반환.
#     ref가 None이면 오늘 기준.
#     """
#     if ref is None:
#         ref = datetime.today()
#     d = ref
#     while True:
#         ymd = d.strftime("%Y%m%d")
#         if is_trading_day(ymd):
#             return ymd
#         d -= timedelta(days=1)

# # 시작일과 종료일 자동 설정   
# Duration = 365 * 2  # 데이터 기간 (일수)     
# End_Date = last_trading_day()
# Start_Date = (datetime.strptime(End_Date, "%Y%m%d") - timedelta(days=Duration)).strftime("%Y%m%d")

@dataclass
class SplitConfig:
    train_ratio: float = 0.7
    val_ratio: float = 0.15  # test_ratio는 1 - train - val
    test_ratio: float = 0.15
    shuffle: bool = False    # 시계열이라 보통 False 권장
    
@dataclass
class DSConfig:
    name: str = ""       # 종목 이름
    code: str = ""       # 종목 코드 (자동설정)
    
    duration: int = 365 * 2  # 데이터 기간 (일수)
    start_date: str = ""  # 자동 결정 (예: "20220101")
    end_date: str = ""    # 자동 결정 (예: "20231231")
    
    # 데이터 관련 설정
    lookback: int = 30     # 과거 데이터 길이 (일수)
    horizons: List[int] = (1, 2, 5)  # 예측 시점 (1일, 2일, 5일 뒤)
    target_kind: str = "logr"   # 타겟 종류: "logr", "pct", "close"
    
    ## 분할 설정    
    #split: SplitConfig = field(default_factory=SplitConfig)
    batch_size: int = 32  # 배치 크기
    # 저장 경로
    dataset_dir: str = "./datasets"  # 데이터셋 저장 디렉토리
    getdata_dir: str = "./csvdata"
    model_dir: str = "./models"  # 모델 저장 디렉토리
    output_dir: str = "./outputs"  # 출력 결과 저장 디렉토리

config = DSConfig()

@dataclass
class FeatureConfig:
    date_cols: List[str] = ("date")
    price_cols: List[str] = ("open", "high", "low", "close", "volume", "chg_pct")
    flow_cols: List[str] = ("inst_sum", "inst_ext", "retail", "foreign")
    fund_cols: List[str] = ("per", "pbr", "div")
    global_cols: List[str] = ("KOSPI", "KOSDAQ", "SN500", "NASDAQ", "eps", "dps")

feature = FeatureConfig()
