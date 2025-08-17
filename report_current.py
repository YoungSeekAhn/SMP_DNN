
import matplotlib
matplotlib.use("TkAgg")   # 또는 "QtAgg" (PyQt5/PySide6 설치 필요)
import matplotlib.pyplot as plt

def plot_investor_ma20(merge_df, investor_cols=["개인", "기관합계", "외국인합계"]):
    """
    수급주체별 20일 이동평균을 플로팅
    investor_cols : merge_df 안에서 수급주체별 컬럼명 리스트
    """
    plt.figure(figsize=(14,6))
    
    for col in investor_cols:
        if col in merge_df.columns:
            plt.plot(
                merge_df.index, 
                merge_df[col].rolling(20).mean(), 
                label=f"{col} 20MA"
            )

    plt.title("Investor Trading Value (20-day Moving Average)")
    plt.xlabel("Date")
    plt.ylabel("Trading Value")
    plt.legend()
    plt.grid(True)
    plt.show()

# 사용 예시

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
    
    