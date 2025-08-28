
import matplotlib
matplotlib.use("TkAgg")   # 또는 "QtAgg" (PyQt5/PySide6 설치 필요)
import matplotlib.pyplot as plt
import os
from pathlib import Path
import pandas as pd
from old.DSConfig import DSConfig


# ---- Figure 그리기 ----
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

cfg = DSConfig()

get_dir = Path(cfg.getdata_dir)
filepath = os.path.join(get_dir, f"{cfg.name}({cfg.code})_{cfg.end_date}.csv")
df = pd.read_csv(filepath, index_col=0, parse_dates=True)

# (1) 주가
close = df['close']
if close.empty:
    raise ValueError("Close price data is empty. Please check the data source.")
axes[0].plot(close.index, close, label="close", color="black")
axes[0].plot(close.rolling(1).mean(), label="MA1", linestyle="--")
axes[0].plot(close.rolling(5).mean(), label="MA5", linestyle="--")
axes[0].plot(close.rolling(20).mean(), label="MA20", linestyle="--")
axes[0].plot(close.rolling(60).mean(), label="MA60", linestyle="--")

axes[0].set_title(f"{cfg.name} ({cfg.code}) 종가 with Moving Averages")
axes[0].set_xlabel("Date")
axes[0].set_ylabel("Price")
axes[0].legend()
axes[0].grid(True)

# (2) 수급주체

#수급주체별 20일 이동평균을 플로팅
#investor_cols : merge_df 안에서 수급주체별 컬럼명 리스트
columns = ['inst_sum', 'inst_ext', 'retail', 'foreign']
Investor_df = df[columns]

for col in Investor_df.columns:
    if col in Investor_df.columns:
        axes[1].plot(
            df.index, 
            df[col].rolling(20).mean(), 
            label=f"{col} 20MA"
        )

axes[1].set_title(f"{cfg.name} ({cfg.code}) Investor Trading Value (20-day Moving Average)")
axes[1].set_ylabel("Trading Value")
axes[1].legend()
axes[1].grid(True)


# (3) PER / PBR
per = df['per']
pbr = df['pbr']
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
    
    