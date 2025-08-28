# backtest_lstm_predictions.py
import numpy as np
import pandas as pd
import os
from pathlib import Path
import matplotlib.pyplot as plt
from dataclasses import dataclass
from old.DSConfig import DSConfig, config

cfg = config

# --------- 설정 ---------
TRADING_DAYS = 252  # 연 환산용
DEFAULT_COST_BPS = 5

# --------- 유틸(안전 평균/지표) ---------
def _nanmean(a):
    a = np.asarray(a, float)
    with np.errstate(invalid="ignore"):
        return np.nan if np.all(np.isnan(a)) else np.nanmean(a)

def rmse(y, yhat):
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    m = np.isfinite(y) & np.isfinite(yhat)
    if not m.any(): return np.nan
    return float(np.sqrt(np.mean((y[m]-yhat[m])**2)))

def mae(y, yhat):
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    m = np.isfinite(y) & np.isfinite(yhat)
    if not m.any(): return np.nan
    return float(np.mean(np.abs(y[m]-yhat[m])))

def mape(y, yhat):
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    m = np.isfinite(y) & np.isfinite(yhat) & (y != 0)
    if not m.any(): return np.nan
    return float(np.mean(np.abs((yhat[m]-y[m]) / y[m])))

def hit_rate(ret_true, ret_pred):
    m = np.isfinite(ret_true) & np.isfinite(ret_pred)
    if not m.any(): return np.nan
    return float(np.mean(np.sign(ret_true[m]) == np.sign(ret_pred[m])))

# --------- 전략 시뮬 ---------
@dataclass
class StrategyConfig:
    cost_bps: float = DEFAULT_COST_BPS
    long_only: bool = False
    threshold: float = 0.0  # long-only 시 진입 임계치 (예: 0.001=0.1%)

def backtest_horizon(df: pd.DataFrame, h: int, cfg: StrategyConfig):
    """
    df: columns ['date', 'true', f'pred_h{h}'] 가정. date는 datetime 가능.
    h: horizon (영업일 수 기준으로 시프트된 테이블이라고 가정)
    cfg: 전략 설정
    반환: metrics(dict), equity(pd.Series), trades(pd.DataFrame)
    """
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.dropna(subset=["date"]).sort_values("date").set_index("date")

    pred_col = f"pred_h{h}"
    if pred_col not in d.columns:
        raise KeyError(f"{pred_col} not in DataFrame columns.")

    # 가격 정확도(타깃일 기준): pred_hh vs true
    price_rmse = rmse(d["true"], d[pred_col])
    price_mae  = mae(d["true"], d[pred_col])
    price_mape = mape(d["true"], d[pred_col])

    # 트레이딩 관점의 진입가: t(행)의 예측은 t-h 시점에 생성 → entry = true.shift(h)
    d[f"entry_h{h}"] = d["true"].shift(h)

    # 수익률(예측/실제)
    d[f"ret_pred_h{h}"]  = d[pred_col] / d[f"entry_h{h}"] - 1.0
    d[f"ret_true_h{h}"]  = d["true"]    / d[f"entry_h{h}"] - 1.0

    # 유효 데이터 마스크
    m = np.isfinite(d[f"entry_h{h}"]) & np.isfinite(d[pred_col]) & np.isfinite(d["true"]) & (d[f"entry_h{h}"]>0)
    d_valid = d.loc[m].copy()

    # 방향 적중률 (수익률 방향 비교)
    dir_acc = hit_rate(d_valid[f"ret_true_h{h}"], d_valid[f"ret_pred_h{h}"])

    # 포지션 생성
    if cfg.long_only:
        d_valid["pos"] = (d_valid[f"ret_pred_h{h}"] > cfg.threshold).astype(float)
    else:
        d_valid["pos"] = np.sign(d_valid[f"ret_pred_h{h}"]).astype(float)  # -1, 0, +1

    # 거래 발생 여부(포지션 변동으로 비용 차감하려면 랙이 필요하지만,
    # 여기서는 간단히 각 트레이드에 일괄 비용 차감: 실현수익에서 cost*|pos| 차감)
    cost = cfg.cost_bps * 1e-4
    d_valid["gross_ret"] = d_valid["pos"] * d_valid[f"ret_true_h{h}"]
    d_valid["net_ret"]   = d_valid["gross_ret"] - cost * d_valid["pos"].abs()

    # 에쿼티 커브
    equity = (1.0 + d_valid["net_ret"]).cumprod()
    total_ret = equity.iloc[-1] - 1.0 if len(equity) else np.nan

    # 연환산 샤프(대략, 일수 기준)
    if len(d_valid) > 1:
        avg = _nanmean(d_valid["net_ret"])
        vol = np.nanstd(d_valid["net_ret"], ddof=1)
        sharpe = (avg / vol) * np.sqrt(TRADING_DAYS) if (vol and vol > 0) else np.nan
    else:
        sharpe = np.nan

    metrics = {
        "h": h,
        "N_predictions": int(m.sum()),
        "Price_RMSE": price_rmse,
        "Price_MAE": price_mae,
        "Price_MAPE": price_mape,
        "Directional_Accuracy": dir_acc,
        "Total_Return": float(total_ret) if np.isfinite(total_ret) else np.nan,
        "Avg_Daily_NetRet": _nanmean(d_valid["net_ret"]),
        "Vol_Daily_NetRet": float(np.nanstd(d_valid["net_ret"], ddof=1)),
        "Sharpe_Ann": sharpe,
    }

    trades = d_valid[[f"entry_h{h}", pred_col, "true", f"ret_pred_h{h}", f"ret_true_h{h}", "pos", "net_ret"]].copy()
    trades.columns = ["entry_price", "pred_price", "actual_price", "pred_ret", "actual_ret", "pos", "net_ret"]

    return metrics, equity, trades

def backtest_all(df: pd.DataFrame, horizons=(1,2,5), cfg: StrategyConfig = StrategyConfig()):
    results = {}
    equities = []
    for h in horizons:
        try:
            m, eq, tr = backtest_horizon(df, h, cfg)
            results[h] = {"metrics": m, "equity": eq, "trades": tr}
            equities.append(eq.rename(f"h={h}"))
        except Exception as e:
            print(f"[h={h}] backtest skipped: {e}")

    # 요약 표
    if results:
        rows = [results[h]["metrics"] for h in sorted(results)]
        summary = pd.DataFrame(rows).set_index("h").sort_index()
        print("\n=== Backtest Summary ===")
        print(summary.to_string(float_format=lambda x: f"{x:,.6f}"))
    else:
        summary = pd.DataFrame()

    # 에쿼티 커브 플롯
    if equities:
        ax = pd.concat(equities, axis=1).plot(figsize=(12,5), title="Equity Curves by Horizon")
        ax.set_xlabel("Date"); ax.set_ylabel("Equity"); ax.grid(True)
        plt.tight_layout(); plt.show()

    return results, summary

# ------------- 사용 예 -------------
if __name__ == "__main__":
    # CSV 예: 질문에 보여주신 wide 포맷
    # columns: date,true,pred_h1,pred_h2,pred_h5, ... (true_h* 컬럼은 없어도 됨)
    code = cfg.code  # 삼성전자
    out_dir = Path(cfg.output_dir)
    
    out_path = out_dir / f"pred_true_wide_{code}.csv"
    
    
    df = pd.read_csv(out_path)
    print(f"Loaded data: {df.shape[0]} rows from {out_path}")
    # long-short, 5bps 비용
    st_cfg = StrategyConfig(cost_bps=5, long_only=False, threshold=0.0)
    results, summary = backtest_all(df, horizons=cfg.horizons, cfg=st_cfg)

    print(results)
    # long-only with threshold (예: 예측수익률이 0.2% 이상일 때만 매수)
    # cfg2 = StrategyConfig(cost_bps=5, long_only=True, threshold=0.002)
    # results2, summary2 = backtest_all(df, horizons=(1,2,5), cfg=cfg2)
