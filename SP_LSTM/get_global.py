import pandas as pd
import yfinance as yf
import datetime
# -----------------------------
# 1) 지수 티커 정의 (Yahoo Finance)
# -----------------------------
def get_global_stock_indices(cfg):
    """
    글로벌 지수 다운로드 후:
      - 미국 지수(S&P500/NASDAQ/SOX)는 전일 값으로 시프트
      - KOSPI가 있는 날짜만 인덱스로 사용하여 정렬
      - 중국/홍콩/일본 지수는 같은 날 값 사용(시프트 없음)
    반환: columns = ['date','KOSPI','SP500_US(-1)','NASDAQ_US(-1)','SOX_US(-1)','SHANGHAI','HANGSENG','NIKKEI']
    """
    TICKERS = {
        "KOSPI":   "^KS11",
        "SP500":   "^GSPC",
        "NASDAQ":  "^IXIC",
        "SOX":     "^SOX",
        "SHANGHAI":"000001.SS",
        "HANGSENG":"^HSI",
        "NIKKEI":  "^N225",
    }

    start = datetime.datetime.strptime(cfg.start_date, "%Y%m%d")
    end   = datetime.datetime.strptime(cfg.end_date,   "%Y%m%d")

    def dl_adj_close(ticker, start, end):
        df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
        if df.empty:
            raise ValueError(f"No data for {ticker}")
        out = df[["Adj Close"]].rename(columns={"Adj Close": ticker})
        out.index = pd.to_datetime(out.index, errors="coerce").tz_localize(None)
        out = out[~out.index.duplicated(keep="last")].sort_index()
        return out

    # 1) 다운로드 후 합치기
    parts = []
    for name, yft in TICKERS.items():
        df = dl_adj_close(yft, start, end).rename(columns={yft: name})
        parts.append(df)
    raw = pd.concat(parts, axis=1, join="outer").sort_index()

    # 2) 항상 1D Series로 꺼내는 헬퍼
    def _as_series(df: pd.DataFrame, col: str) -> pd.Series:
        if col not in df.columns:
            return pd.Series(index=df.index, dtype="float64", name=col)
        s = df[col]
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        return pd.to_numeric(s, errors="coerce")

    # 3) 시리즈 추출(+미국지수 shift)
    kospi    = _as_series(raw, "KOSPI")
    sp500    = _as_series(raw, "SP500").shift(1)
    nasdaq   = _as_series(raw, "NASDAQ").shift(1)
    sox      = _as_series(raw, "SOX").shift(1)
    shanghai = _as_series(raw, "SHANGHAI")
    hangseng = _as_series(raw, "HANGSENG")
    nikkei   = _as_series(raw, "NIKKEI")

    # 4) KOSPI 있는 날만 인덱스
    mask = kospi.notna().to_numpy()
    kospi_days = raw.index[mask]

    # 5) 리샘플/정렬
    aligned = pd.DataFrame(index=kospi_days)
    aligned["KOSPI"]         = kospi.reindex(kospi_days)
    aligned["SP500_US(-1)"]  = sp500.reindex(kospi_days)
    aligned["NASDAQ_US(-1)"] = nasdaq.reindex(kospi_days)
    aligned["SOX_US(-1)"]    = sox.reindex(kospi_days)
    aligned["SHANGHAI"]      = shanghai.reindex(kospi_days)
    aligned["HANGSENG"]      = hangseng.reindex(kospi_days)
    aligned["NIKKEI"]        = nikkei.reindex(kospi_days)
    aligned = aligned.sort_index()

    # 6) 'date' 컬럼 추가(인덱스 → 컬럼), tz-naive 보장
    aligned = aligned.reset_index()
    # 인덱스 이름이 없으면 'index'로 들어오므로 안전 리네임
    first_col = aligned.columns[0]
    if first_col != "date":
        aligned = aligned.rename(columns={first_col: "date"})
    aligned["date"] = pd.to_datetime(aligned["date"], errors="coerce").dt.tz_localize(None)

    # (선택) 열 순서 정리: date 먼저
    cols = ["date"] + [c for c in aligned.columns if c != "date"]
    aligned = aligned[cols]

    return aligned
