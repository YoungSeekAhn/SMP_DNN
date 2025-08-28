# -*- coding: utf-8 -*-
"""
score_universe.py
- KOSPI 시총 상위 100개 종목을 대상으로 조건별 점수 산정
- 기술적(6), 거래량(2), 수급(5) = 총 13개 항목 → 최대 100점
- 총점 상위 5개 종목 출력 + 상세 점수 브레이크다운

필요 패키지:
  pip install finance-datareader pykrx pandas numpy
"""

import warnings
warnings.filterwarnings("ignore")

import datetime as dt
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import FinanceDataReader as fdr
from pykrx import stock as krx


# ===================== 설정 =====================
TOP_N_MARCAP = 100          # 시총 상위 N
PICK_TOP = 5                # 최종 선별 수
INDEX_TICKER = "KS11"       # KOSPI 지수 (FinanceDataReader)
LOOKBACK_DAYS = 400         # 지표 계산기간(영업일 여유 포함)

# 점수 테이블(요청안)
WEIGHTS = {
    # --- 기술적 ---
    "rsi30": 10,                # (1) RSI < 30
    "momentum_pos": 10,         # (2) MOM>0
    "macd_cross": 10,           # (3) MACD 골든크로스
    "ema5_over_ema20": 5,       # (4) 5일선이 20일선 돌파
    "ema20_over_ema60": 10,     # (5) 20일선이 60일선 돌파
    "rs_plus": 10,              # (6) 20일 상대강도(종목-KOSPI) +

    # --- 거래량 ---
    "vol_120": 5,               # (7) 3일평균 / 20일평균 ≥ 1.2
    "vol_150": 10,              # (8) 3일평균 / 20일평균 ≥ 1.5

    # --- 수급 ---
    "frg_own_1m_up": 10,        # (9) 외국인 지분율 1개월 증가
    "frg_3_pos": 5,             # (10) 외국인 3일 연속 순매수 증가(평균 대비 +)
    "frg_5_pos": 5,             # (11) 외국인 5일 연속 순매수 증가
    "ins_3_pos": 5,             # (12) 기관 3일 연속 순매수 증가
    "ins_5_pos": 5,             # (13) 기관 5일 연속 순매수 증가
}

# 거래량 점수 합산 정책: 120%와 150%가 동시 충족 시
VOLUME_SCORE_MODE = "max"  # "max"=상한만 채택, "sum"=둘 다 더함
# =================================================


# ===================== 유틸/헬퍼 =====================
def add_indicators(px: pd.DataFrame) -> pd.DataFrame:
    """
    OHLCV 데이터프레임(px)에 기술적 지표 추가:
      - RSI(14), MOM10, EMA5/20/60, MACD(12,26,9), VOL_3MA, VOL_MA20
    """
    out = px.copy()
    # 컬럼 표준화
    rename_map = {c: c.capitalize() for c in out.columns}
    out = out.rename(columns=rename_map)
    for need in ["Open", "High", "Low", "Close", "Volume"]:
        if need not in out.columns:
            raise ValueError(f"'{need}' column missing in price DataFrame")

    # RSI(14)
    delta = out["Close"].diff()
    up = delta.clip(lower=0.0).rolling(14).mean()
    down = (-delta.clip(upper=0.0)).rolling(14).mean()
    rs = up / (down + 1e-12)
    out["RSI14"] = 100 - (100 / (1 + rs))

    # Momentum(10)
    out["MOM10"] = out["Close"] - out["Close"].shift(10)
    out["MOM5"] = out["Close"] - out["Close"].shift(5)

    # EMA
    out["EMA5"] = out["Close"].ewm(span=5, adjust=False).mean()
    out["EMA20"] = out["Close"].ewm(span=20, adjust=False).mean()
    out["EMA60"] = out["Close"].ewm(span=60, adjust=False).mean()

    # MACD (12, 26, 9)
    ema12 = out["Close"].ewm(span=12, adjust=False).mean()
    ema26 = out["Close"].ewm(span=26, adjust=False).mean()
    out["MACD"] = ema12 - ema26
    out["MACD_SIG"] = out["MACD"].ewm(span=9, adjust=False).mean()

    # 거래량 이동평균
    out["VOL_3MA"] = out["Volume"].rolling(3).mean()
    out["VOL_MA20"] = out["Volume"].rolling(20).mean()

    return out


def fetch_investor_netbuy_df(ticker: str, start_yyyymmdd: str, end_yyyymmdd: str) -> pd.DataFrame:
    """
    투자자별 순매수(금액) – pykrx
    반환: index=datetime, columns=['FRG','INS']  (원 단위, +면 순매수)
    """
    # 티커 zero-padding 보정 (예: '5930' -> '005930')
    if isinstance(ticker, str) and ticker.isdigit() and len(ticker) < 6:
        ticker = ticker.zfill(6)

    # pykrx: 일자별 투자자 거래대금(순매수; 기본) DataFrame
    df = krx.get_market_trading_value_by_date(start_yyyymmdd, end_yyyymmdd, ticker)
    if df is None or df.empty:
        return pd.DataFrame(columns=["FRG", "INS"])

    # 날짜 인덱스 정리
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
    df = df.sort_index()

    # 컬럼 가드 (버전/시장/종목에 따라 일부 열 부재 가능)
    col_map = {
        "FRG": ["외국인", "외국인합계"],     # 상황에 따라 컬럼명이 다를 수 있음
        "INS": ["기관합계", "기관"]          # 기관합계가 일반적
    }

    def pick_col(frame: pd.DataFrame, candidates: list[str]) -> pd.Series:
        for c in candidates:
            if c in frame.columns:
                s = pd.to_numeric(frame[c], errors="coerce")
                s.name = None
                return s
        # 없으면 빈 시리즈 반환
        return pd.Series(index=frame.index, dtype=float)

    s_frg = pick_col(df, col_map["FRG"])
    s_ins = pick_col(df, col_map["INS"])

    out = pd.concat([s_frg.rename("FRG"), s_ins.rename("INS")], axis=1)

    # NaN 전부면 빈 DF 반환
    if out["FRG"].isna().all() and out["INS"].isna().all():
        return pd.DataFrame(columns=["FRG", "INS"])

    return out


def foreign_ownership_ratio(date_yyyymmdd: str, market="KOSPI") -> pd.DataFrame:
    """
    외국인 보유비중/한도소진율 등 지표 – 버전마다 컬럼명이 다를 수 있어 추론.
    반환: ['티커','지분율'] 로 정규화
    """
    try:
        df = krx.get_exhaustion_rates_of_foreign_investment_by_ticker(date_yyyymmdd, market=market)
        if "티커" not in df.columns:
            df = df.reset_index().rename(columns={"index": "티커"})
        cand = [c for c in df.columns if ("보유" in c) or ("소진" in c) or ("외국" in c)]
        col = cand[0] if cand else df.columns[-1]
        return df[["티커", col]].rename(columns={col: "지분율"})
    except Exception:
        return pd.DataFrame(columns=["티커", "지분율"])


# ===================== 점수 계산 =====================
@dataclass
class ScoreBreakdown:
    # 기술적
    rsi30: int = 0
    momentum_pos: int = 0
    macd_cross: int = 0
    ema5_over_ema20: int = 0
    ema20_over_ema60: int = 0
    rs_plus: int = 0
    # 거래량
    vol_120: int = 0
    vol_150: int = 0
    # 수급
    frg_own_1m_up: int = 0
    frg_3_pos: int = 0
    frg_5_pos: int = 0
    ins_3_pos: int = 0
    ins_5_pos: int = 0

    def total(self) -> int:
        vol_part = max(self.vol_120, self.vol_150) if VOLUME_SCORE_MODE == "max" else (self.vol_120 + self.vol_150)
        # 원 점수 합에서 거래량 부분을 교체
        base_sum = (self.rsi30 + self.momentum_pos + self.macd_cross +
                    self.ema5_over_ema20 + self.ema20_over_ema60 + self.rs_plus +
                    self.frg_own_1m_up + self.frg_3_pos + self.frg_5_pos +
                    self.ins_3_pos + self.ins_5_pos)
        return base_sum + (vol_part)  # vol_120/150는 이미 0으로 초기화되어 있음


def score_one(
    ticker: str,
    px: pd.DataFrame,               # OHLCV + 인디케이터
    inv: pd.DataFrame,              # 투자자 순매수 금액 ['FRG','INS']
    kospi_close: pd.Series,         # KOSPI 종가
    frg_now: float = np.nan,        # 외국인 지분율(오늘 근처)
    frg_1m: float = np.nan          # 외국인 지분율(1개월 전)
) -> Tuple[int, ScoreBreakdown]:
    """
    요청하신 13개 조건으로 점수 산정
    """
    bd = ScoreBreakdown()
    if len(px) < 65:
        return 0, bd
    last = px.index.max()

    # (1) RSI < 30
    try:
        if px.loc[last, "RSI14"] < 30:
            bd.rsi30 = WEIGHTS["rsi30"]
    except Exception:
        pass

    # (2) Momentum > 0 (10일)
    try:
        #if px.loc[last, "MOM10"] > 0:
        if px.loc[last, "MOM5"] > 0:
            bd.momentum_pos = WEIGHTS["momentum_pos"]
    except Exception:
        pass

    # (3) MACD 골든크로스 (MACD > Signal & 직전 <=)
    try:
        macd, sig = px["MACD"], px["MACD_SIG"]
        if (macd.iloc[-1] > sig.iloc[-1]) and (macd.iloc[-2] <= sig.iloc[-2]):
            bd.macd_cross = WEIGHTS["macd_cross"]
    except Exception:
        pass

    # (4) 5일선 > 20일선으로 오늘 돌파
    try:
        now = px["EMA5"].iloc[-1] > px["EMA20"].iloc[-1]
        prev = px["EMA5"].iloc[-2] <= px["EMA20"].iloc[-2]
        if now and prev:
            bd.ema5_over_ema20 = WEIGHTS["ema5_over_ema20"]
    except Exception:
        pass

    # (5) 20일선 > 60일선으로 오늘 돌파
    try:
        now = px["EMA20"].iloc[-1] > px["EMA60"].iloc[-1]
        prev = px["EMA20"].iloc[-2] <= px["EMA60"].iloc[-2]
        if now and prev:
            bd.ema20_over_ema60 = WEIGHTS["ema20_over_ema60"]
    except Exception:
        pass

    # (6) 최근 20일 KOSPI 대비 상대 강도 +
    try:
        common = px.index.intersection(kospi_close.index)
        if len(common) >= 21:
            r_stock = px.loc[common, "Close"].pct_change(20).iloc[-1]
            r_kospi = kospi_close.loc[common].pct_change(20).iloc[-1]
            if (r_stock - r_kospi) > 0:
                bd.rs_plus = WEIGHTS["rs_plus"]
    except Exception:
        pass

    # (7)(8) 거래량: 3일평균 / 20일평균
    try:
        ratio = px.loc[last, "VOL_3MA"] / (px.loc[last, "VOL_MA20"] + 1e-12)
        if ratio >= 1.2:
            bd.vol_120 = WEIGHTS["vol_120"]
        if ratio >= 1.5:
            bd.vol_150 = WEIGHTS["vol_150"]
            if VOLUME_SCORE_MODE == "max":
                bd.vol_120 = 0
    except Exception:
        pass

    # (9) 외국인 지분율 1개월 증가
    try:
        if np.isfinite(frg_now) and np.isfinite(frg_1m) and (frg_now - frg_1m) > 0:
            bd.frg_own_1m_up = WEIGHTS["frg_own_1m_up"]
    except Exception:
        pass

    # (10) 외국인 3일 연속 순매수 증가 (+)
    try:
        if len(inv) >= 3 and (inv["FRG"].iloc[-3:] > 0).all():
            bd.frg_3_pos = WEIGHTS["frg_3_pos"]
    except Exception:
        pass

    # (11) 외국인 5일 연속 순매수 증가
    try:
        if len(inv) >= 5 and (inv["FRG"].iloc[-5:] > 0).all():
            bd.frg_5_pos = WEIGHTS["frg_5_pos"]
    except Exception:
        pass

    # (12) 기관 3일 연속 순매수 증가
    try:
        if len(inv) >= 3 and (inv["INS"].iloc[-3:] > 0).all():
            bd.ins_3_pos = WEIGHTS["ins_3_pos"]
    except Exception:
        pass

    # (13) 기관 5일 연속 순매수 증가
    try:
        if len(inv) >= 5 and (inv["INS"].iloc[-5:] > 0).all():
            bd.ins_5_pos = WEIGHTS["ins_5_pos"]
    except Exception:
        pass

    return bd.total(), bd


# ===================== 메인 파이프라인 =====================
def main():
    # 날짜 설정
    today = dt.date.today()
    start = today - dt.timedelta(days=LOOKBACK_DAYS)
    s_fdr = start.strftime("%Y-%m-%d")
    e_fdr = today.strftime("%Y-%m-%d")
    s_krx = start.strftime("%Y%m%d")
    e_krx = today.strftime("%Y%m%d")
    d1m  = (today - dt.timedelta(days=30)).strftime("%Y%m%d")

    print(f"[기간] {s_fdr} ~ {e_fdr}")

    # 1) KOSPI 시총 상위 100 종목
    kospi = fdr.StockListing("KOSPI")
    kospi = kospi.dropna(subset=["Marcap"]).sort_values("Marcap", ascending=False)
    uni = kospi.head(TOP_N_MARCAP).copy()
    print(f"[유니버스] {len(uni)} 종목")

    # 2) KOSPI 지수 (상대강도용)
    idx_df = fdr.DataReader(INDEX_TICKER, s_fdr, e_fdr)
    if "Close" not in idx_df.columns:
        idx_df.rename(columns={c: c.capitalize() for c in idx_df.columns}, inplace=True)
    kospi_close = idx_df["Close"]

    # 3) 외국인 지분율 맵 (오늘 vs 1개월 전)
    fr_now_df = foreign_ownership_ratio(e_krx, market="KOSPI")
    fr_1m_df  = foreign_ownership_ratio(d1m,  market="KOSPI")
    fr_now_map = dict(zip(fr_now_df["티커"], fr_now_df["지분율"]))
    fr_1m_map  = dict(zip(fr_1m_df["티커"],  fr_1m_df["지분율"]))

    # KOSPI200 구성종목 티커(6자리) 리스트
    tickers = krx.get_index_portfolio_deposit_file("1028")

    # 종목명 붙이기
    df = pd.DataFrame({"Code": tickers})
    uni = uni[uni["Code"].isin(tickers)].copy()
    uni["Marcap"] = uni["Marcap"].astype(float)
    uni["Name"] = np.nan
    
    uni["Name"] = df["Code"].apply(krx.get_market_ticker_name)
    
    rows: List[Dict] = []

    for _, r in uni.iterrows():
        ticker = r["Code"]          # 예: '005930' 형태
        name   = r.get("Name", "")

        try:
            # 4) 가격/거래량 (KRX OHLCV) + 인디케이터
            #  - get_market_ohlcv_by_date는 인덱스가 날짜
            px = krx.get_market_ohlcv_by_date(s_krx, e_krx, ticker)
            if px is None or px.empty:
                continue

            # 컬럼 표준화: 기존 코드가 'Close' 등 영문을 기대했다면 매핑
            # (pykrx 기본: 시가/고가/저가/종가/거래량/등락률)
            px = px.rename(columns={
                "시가": "Open",
                "고가": "High",
                "저가": "Low",
                "종가": "Close",
                "거래량": "Volume"
            })[["Open","High","Low","Close","Volume"]]

            px.index = pd.to_datetime(px.index)
            px = add_indicators(px)  # 기존 함수 재사용

            # 5) 투자자 수급 (이미 KRX 기반이면 그대로, 아니면 pykrx 예시 참고)
            #    예시) 개인/외국인/기관 순매수:
            # inv = stock.get_market_trading_value_by_date(s_krx, e_krx, ticker, detail=True)
            inv = fetch_investor_netbuy_df(ticker, s_krx, e_krx)

            # 6) 외국인 지분율 (오늘 vs 1M 전) - KRX 소진율 API 활용 예시
            #    소진율 데이터: get_exhaustion_rates_of_foreign_investment_by_date
            fr_df = krx.get_exhaustion_rates_of_foreign_investment_by_date(s_krx, e_krx, ticker)
            # 컬럼 예: 보유량, 한도수량, 한도소진율(%)
            if fr_df is None or fr_df.empty:
                frg_now = np.nan
                frg_1m  = np.nan
            else:
                fr_df.index = pd.to_datetime(fr_df.index)
                fr_df = fr_df.sort_index()
                # 오늘(가장 최근) 값
                frg_now = float(fr_df["한도소진율"].iloc[-1]) if "한도소진율" in fr_df.columns else np.nan
                # 1M 전 근사(거래일 기준 22영업일 전)
                if len(fr_df) > 22:
                    frg_1m = float(fr_df["한도소진율"].iloc[-23]) if "한도소진율" in fr_df.columns else np.nan
                else:
                    frg_1m = np.nan

            # 7) 점수 산정
            total, bd = score_one(
                ticker=ticker, px=px, inv=inv, kospi_close=kospi_close,
                frg_now=frg_now, frg_1m=frg_1m
            )

            rows.append({
                "Ticker": ticker,
                "Name": name,
                "Marcap": r["Marcap"],
                "Score": total,
                **bd.__dict__
            })

        except Exception:
            # 휴장/데이터 누락/형식 차이 등은 스킵
            continue

    if not rows:
        print("※ 결과 없음: 네트워크/거래일/pykrx 반환 형식 등을 확인하세요.")
    else:
        out = pd.DataFrame(rows).sort_values(["Score", "Marcap"], ascending=[False, False])

    # 출력
    print("\n=== 총점 상위 5개 종목 ===")
    print(out.head(PICK_TOP)[["Ticker", "Name", "Marcap", "Score"]])

    print("\n=== 점수 브레이크다운 (상위 5개) ===")
    cols = ["Ticker","Name","Score"] + list(WEIGHTS.keys())
    print(out.head(PICK_TOP)[cols])

    # 저장 (선택)
    out.to_csv("universe_scored.csv", index=False, encoding="utf-8-sig")
    print("\n[저장] universe_scored.csv")

if __name__ == "__main__":
    main()
