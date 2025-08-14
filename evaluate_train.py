# plot_test_prices.py
# ------------------------------------------------------------
# 저장된 모델(models/{Code}_best.pt)을 불러와
# 테스트셋에서 h=1 예측을 이용해 "실제 주가 스케일"로 복원/그래프
# ------------------------------------------------------------
import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt

# 1) train_dataset에서 필요한 항목 import
from train_dataset import (
    MultiInputLSTM,  # 모델 클래스
    meta,            # 메타정보 (feature_config, horizons, target_kind 등)
    device,          # torch.device
    test_loader,     # 테스트 DataLoader
    Code             # 종목 코드 (파일명 용)
)

# ===== 유틸: loader에서 입출력 차원 파악 후 동일 아키텍처 구성 =====
def build_model_from_loader(loader, hidden=64, layers=1, head_hidden=128, dropout=0.1):
    sample = loader.dataset[0]
    p_dim = sample["x_price"].shape[-1]
    f_dim = sample["x_flow"].shape[-1]
    d_dim = sample["x_fund"].shape[-1]
    out_dim = sample["y"].shape[-1]  # horizon 개수
    model = MultiInputLSTM(
        p_dim=p_dim, f_dim=f_dim, d_dim=d_dim,
        hidden=hidden, layers=layers,
        head_hidden=head_hidden, out_dim=out_dim,
        dropout=dropout
    ).to(device)
    return model, out_dim

# ===== 유틸: price inverse_transform 시도 (스케일러가 있으면 역변환, 없으면 그대로) =====
def maybe_inverse_close_from_ds(loader, close_idx: int) -> float:
    """
    테스트 첫 배치의 첫 샘플에서 lookback 마지막 시점 close를 가져와
    (가능하면) 스케일 역변환한 값을 초기 P0로 사용.
    """
    ds = loader.dataset
    first_batch = next(iter(loader))
    x_close_scaled = float(first_batch["x_price"][0, -1, close_idx])

    # 스케일러 자동 탐색
    scaler_obj = None
    # 후보 속성명
    candidates = ["scaler_price", "price_scaler", "scaler_price_branch", "scalers", "scaler"]
    for name in candidates:
        if hasattr(ds, name):
            obj = getattr(ds, name)
            # dict/네임스페이스 형태 처리
            if isinstance(obj, dict):
                for k in ["price", "price_branch"]:
                    if k in obj and hasattr(obj[k], "inverse_transform"):
                        scaler_obj = obj[k]
                        break
            elif hasattr(obj, "inverse_transform"):
                scaler_obj = obj
        if scaler_obj is not None:
            break

    # 스케일러 있으면 price-branch 전체 벡터로 inverse → close만 추출
    if scaler_obj is not None:
        try:
            x_price_last_vec = first_batch["x_price"][0, -1].numpy().reshape(1, -1)  # (1, P)
            inv = scaler_obj.inverse_transform(x_price_last_vec).reshape(-1)         # (P,)
            return float(inv[close_idx])
        except Exception:
            pass

    # 스케일러가 없거나 실패하면 스케일되지 않은 것으로 가정
    return x_close_scaled

# ===== 유틸: h=1 로그수익률/수익률로 절대가격 경로 복원 =====
def reconstruct_path_h1(loader, preds: np.ndarray, trues: np.ndarray, close_idx: int, target_kind: str):
    """
    preds, trues: (N, H)
    close_idx: price_cols에서 'close'의 인덱스
    target_kind: 'logr' (로그수익률) 또는 그 외(일반 수익률)
    반환: (t, price_true, price_pred)
    """
    horizons = meta.get("window", {}).get("horizons", [1])
    h_idx = horizons.index(1) if 1 in horizons else 0  # h=1이 없으면 첫 열 사용

    # 초기 가격 P0
    P0 = maybe_inverse_close_from_ds(loader, close_idx)

    r_pred = preds[:, h_idx]  # (N,)
    r_true = trues[:, h_idx]  # (N,)

    if target_kind == "logr":
        growth_pred = np.exp(r_pred)
        growth_true = np.exp(r_true)
    else:
        growth_pred = 1.0 + r_pred
        growth_true = 1.0 + r_true

    price_pred = [P0]
    price_true = [P0]
    for gp in growth_pred:
        price_pred.append(price_pred[-1] * gp)
    for gt in growth_true:
        price_true.append(price_true[-1] * gt)

    t = np.arange(len(price_true))  # 길이 N+1
    return t, np.array(price_true), np.array(price_pred)

def main():
    # 0) 체크포인트 경로
    ckpt_path = Path("models") / f"{Code}_best.pt"
    assert ckpt_path.exists(), f"체크포인트 파일이 없습니다: {ckpt_path}"

    # 1) 모델 구성 & 로드
    model, _ = build_model_from_loader(test_loader)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    target_kind = ckpt.get("target_kind", meta.get("target_kind", "logr"))

    # 2) 테스트셋 예측 수집
    Ys, Ps = [], []
    with torch.no_grad():
        for b in test_loader:
            p = model(b["x_price"].to(device),
                      b["x_flow"].to(device),
                      b["x_fund"].to(device))
            Ys.append(b["y"])       # CPU 텐서
            Ps.append(p.cpu())      # CPU 텐서
    Y = torch.cat(Ys, dim=0).numpy()  # (N, H)
    P = torch.cat(Ps, dim=0).numpy()  # (N, H)

    # 3) close 인덱스 확인
    price_cols = meta.get("feature_config", {}).get("price_cols", [])
    try:
        close_idx = price_cols.index("close")
    except ValueError:
        close_idx = 3  # open, high, low, close, ... 가정
        print("[WARN] meta.feature_config.price_cols에서 'close'를 찾지 못해 close_idx=3으로 가정합니다.")

    # 4) 절대가격 경로 복원 (h=1 기준)
    t, price_true, price_pred = reconstruct_path_h1(test_loader, P, Y, close_idx, target_kind)

    # 5) 그래프
    plt.figure(2, figsize=(12,5))
    plt.plot(t, price_true, label="Actual (reconstructed)", linewidth=2)
    plt.plot(t, price_pred, label="Predicted (reconstructed)", alpha=0.85)
    plt.title(f"Test: Actual vs Predicted Price Path (Code={Code}, h=1)")
    plt.xlabel("Test steps")
    plt.ylabel("Price")
    plt.grid(True); plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()