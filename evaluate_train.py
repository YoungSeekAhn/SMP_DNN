# plot_test_prices.py
# ------------------------------------------------------------
# 저장된 모델(models/{Code}_best.pt)을 불러와
# 테스트셋에서 h=1 예측을 이용해 "실제 주가 스케일"로 복원/그래프
# ------------------------------------------------------------
import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import pickle

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

    # 1) scalers 로드

    with open("artifacts/scalers.pkl", "rb") as f:
        scalers_loaded = pickle.load(f)

    # 2) P0 추출 (역스케일 포함)
    P0 = initial_close_from_loader(test_loader, scalers_loaded, meta["feature_config"])

    # 3) 테스트 예측/정답 수집
    Ys, Ps = [], []
    
    with torch.no_grad():
        for b in test_loader:
            p = model(b["x_price"].to(device),
                    b["x_flow"].to(device),
                    b["x_fund"].to(device))
            Ys.append(b["y"]); Ps.append(p.cpu())
    Y = torch.cat(Ys, dim=0).numpy()  # (N, H)
    P = torch.cat(Ps, dim=0).numpy()  # (N, H)

    # 4) 절대 가격 경로 복원 및 플롯
    horizons = meta["window"]["horizons"]
    target_kind = meta.get("target_kind", "logr")
    t, price_true, price_pred = reconstruct_abs_path_h1(P, Y, P0, horizons, target_kind)
    plt.figure(figsize=(12,5))
    plt.plot(t, price_true, label="Actual (reconstructed)", linewidth=2)
    plt.plot(t, price_pred, label="Predicted (reconstructed)", alpha=0.85)
    plt.title("Test: Absolute Price Path (h=1)")
    plt.xlabel("Test steps"); plt.ylabel("Price")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.show()


def get_close_idx_from_feat(feat):
    price_cols = feat.get("price_cols", [])
    try:
        return price_cols.index("close")
    except ValueError:
        # open, high, low, close, volume, ... 라면 보통 close=3
        return 3

def initial_close_from_loader(test_loader, scalers, feat):
    """
    테스트 첫 배치 첫 샘플의 lookback 마지막 시점에서 close를 꺼내어,
    price 스케일러가 있으면 inverse_transform하여 P0(절대가격) 반환.
    """
    close_idx = get_close_idx_from_feat(feat)
    first_batch = next(iter(test_loader))
    # x_price shape: (B, L, P)
    last_price_vec = first_batch["x_price"][0, -1].cpu().numpy()  # (P,)
    sc_price = getattr(scalers, "price", None)
    if sc_price is not None and hasattr(sc_price, "inverse_transform"):
        inv = sc_price.inverse_transform(last_price_vec.reshape(1, -1)).reshape(-1)
        return float(inv[close_idx])
    return float(last_price_vec[close_idx])  # 스케일 안된 경우

def reconstruct_abs_path_h1(preds, trues, P0, horizons, target_kind="logr"):
    """
    preds, trues: (N, H) numpy 배열
    P0: 초기 절대가격 (float)
    horizons: 예: [1,2,3]
    target_kind: 'logr'이면 exp, 그 외(수익률)면 (1+r) 사용
    """
    h_idx = horizons.index(1) if 1 in horizons else 0
    r_pred = preds[:, h_idx]
    r_true = trues[:, h_idx]

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

    t = np.arange(len(price_true))
    return t, np.array(price_true), np.array(price_pred)


if __name__ == "__main__":
    main()