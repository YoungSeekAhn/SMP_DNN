# DSConfig.py
from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime, timedelta

@dataclass
class SplitConfig:
    train_ratio: float = 0.7
    val_ratio: float = 0.15  # test_ratio는 1 - train - val
    shuffle: bool = False    # 시계열이라 보통 False 권장

@dataclass
class DSConfig:
    # 필수
    code: str                 # 예: "005930" (삼성전자)
    lookback: int             # 예: 60
    horizons: List[int]       # 예: [1, 5, 20, 60]
    target_kind: str          # "close" or "return"

    # 데이터 기간 (YYYYMMDD). None이면: Start=오늘 기준 마지막 거래일, End=2년 전
    start_date: Optional[str] = None
    end_date: Optional[str] = None

    # 스플릿
    split: SplitConfig = field(default_factory=SplitConfig)

    # 저장 경로
    out_dir: str = "./artifacts"

    def ensure_dates(self):
        """start_date(최근 거래일), end_date(2년 전) 자동 보정"""
        if self.start_date is None or self.end_date is None:
            from pykrx import stock
            from datetime import datetime, timedelta

            # 마지막 거래일 찾기 (지수 1001로 확인해도 됨)
            def is_trading_day(ymd: str, ticker: str = "005930") -> bool:
                df = stock.get_market_ohlcv_by_date(ymd, ymd, ticker)
                return df is not None and len(df) > 0

            d = datetime.today()
            while True:
                ymd = d.strftime("%Y%m%d")
                if is_trading_day(ymd, self.code):
                    last = ymd
                    break
                d -= timedelta(days=1)

            two_years_ago = (datetime.strptime(last, "%Y%m%d") - timedelta(days=365*2)).strftime("%Y%m%d")

            if self.start_date is None:
                self.start_date = two_years_ago
                self.end_date   = last
            elif self.end_date is None:
                self.end_date = last
