from dataset_functions import add_multi_targets, time_split, auto_feature_config, fit_scalers, apply_scalers, build_windows
from dataset_functions import MultiInputTSDataset, PipelineConfig, SplitConfig
import pandas as pd
from pathlib import Path
import os

Code = "005930"
out_dir = Path.cwd()
filepath = os.path.join(out_dir, f"{Code}_merged_data.csv")
merged = pd.read_csv(filepath, index_col=0, parse_dates=True)

# target kind can be "logr", "pct", or "close"
cfg = PipelineConfig()
cfg.lookback = 20 # Lookback period for the model
cfg.horizons = [1, 2, 3] # Prediction horizon (3 days ahead)
cfg.target_kind = "logr"  # or "pct", or "close"
cfg.split = SplitConfig(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)


# 2) target
with_target = add_multi_targets(merged, horizons=cfg.horizons, target_kind=cfg.target_kind)

# 3) split
tr, va, te = time_split(with_target, cfg.split)

# 4) features
feat = auto_feature_config(with_target, cfg)

# 5) scalers fit on train, apply to all
scalers = fit_scalers(tr, feat)
tr_s = apply_scalers(tr, scalers, feat)
va_s = apply_scalers(va, scalers, feat)
te_s = apply_scalers(te, scalers, feat)

# 6) windows
Xp_tr, Xf_tr, Xd_tr, y_tr = build_windows(tr_s, feat, lookback=cfg.lookback, horizons=cfg.horizons)
Xp_va, Xf_va, Xd_va, y_va = build_windows(va_s, feat, lookback=cfg.lookback, horizons=cfg.horizons)
Xp_te, Xf_te, Xd_te, y_te = build_windows(te_s, feat, lookback=cfg.lookback, horizons=cfg.horizons)

# 7) torch datasets
ds_tr = MultiInputTSDataset(Xp_tr, Xf_tr, Xd_tr, y_tr)
ds_va = MultiInputTSDataset(Xp_va, Xf_va, Xd_va, y_va)
ds_te = MultiInputTSDataset(Xp_te, Xf_te, Xd_te, y_te)
meta = {
    "feature_config": feat.__dict__,
    "scalers": "StandardScaler per-branch (fit on train)",
    "splits": cfg.split.__dict__,
    "window": {"lookback": cfg.lookback, "horizons": cfg.horizons},
    "target_kind": cfg.target_kind,
    "lengths": {"train": len(ds_tr), "val": len(ds_va), "test": len(ds_te)},
}
# Save the dataset and metadata
dataset_dir = out_dir / "datasets"
dataset_dir.mkdir(exist_ok=True, parents=True)  # Create the directory if it doesn't exist  
dataset_path = dataset_dir / f"{Code}_dataset.pkl"

# Save the dataset and metadata
with open(dataset_path, "wb") as f:
    pd.to_pickle({
        "train": ds_tr,
        "val": ds_va,
        "test": ds_te,
        "meta": meta
    }, f)
    
print(f"Dataset saved to {dataset_path}")
print("Metadata:", meta)

