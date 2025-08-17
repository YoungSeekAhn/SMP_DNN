# evaluate_train.py
import math
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# ===== dataset 로드 (make_dataset.py에서 저장한 배열 사용) =====
def load_payload(code):
    payload_path = Path("datasets") / f"{code}_dataset.pkl"
    assert payload_path.exists(), f"payload not found: {payload_path}"
    return pd.read_pickle(payload_path)

class MIDataset(Dataset):
    def __init__(self, Xp, Xf, Xd, Y): self.Xp,self.Xf,self.Xd,self.Y = Xp,Xf,Xd,Y
    def __len__(self): return len(self.Y)
    def __getitem__(self, i):
        return {
            "x_price": torch.from_numpy(self.Xp[i]),
            "x_flow":  torch.from_numpy(self.Xf[i]),
            "x_fund":  torch.from_numpy(self.Xd[i]),
            "y":       torch.from_numpy(self.Y[i]),
        }

# ===== 모델 정의(동일 구조) =====
class BranchLSTM(nn.Module):
    def __init__(self, in_dim, hidden=96, layers=1, dropout=0.2):
        super().__init__()
        self.in_dim=in_dim
        if in_dim>0:
            self.lstm=nn.LSTM(in_dim,hidden,num_layers=layers,batch_first=True,dropout=(dropout if layers>1 else 0.0))
            self.out_dim=hidden
        else:
            self.lstm=None; self.out_dim=0
    def forward(self,x):
        if self.in_dim==0: return x.new_zeros(x.size(0),0)
        _,(h,_) = self.lstm(x); return h[-1]

class MultiInputLSTM(nn.Module):
    def __init__(self,p_dim,f_dim,d_dim,hidden=96,layers=1,head_hidden=128,out_dim=1,dropout=0.2):
        super().__init__()
        self.enc_p=BranchLSTM(p_dim,hidden,layers,dropout)
        self.enc_f=BranchLSTM(f_dim,hidden,layers,dropout)
        self.enc_d=BranchLSTM(d_dim,hidden,layers,dropout)
        cat=self.enc_p.out_dim+self.enc_f.out_dim+self.enc_d.out_dim
        self.mlp=nn.Sequential(nn.Linear(cat,head_hidden),nn.ReLU(),nn.Dropout(dropout),nn.Linear(head_hidden,out_dim))
    def forward(self,xp,xf,xd):
        h=torch.cat([self.enc_p(xp),self.enc_f(xf),self.enc_d(xd)],dim=-1)
        return self.mlp(h)

def evaluate(model, loader, device, target_kind="logr"):
    criterion = nn.MSELoss()
    model.eval(); total=0; n=0; Ys=[]; Ps=[]
    with torch.no_grad():
        for b in loader:
            xp=b["x_price"].to(device); xf=b["x_flow"].to(device); xd=b["x_fund"].to(device); y=b["y"].to(device)
            p=model(xp,xf,xd); loss=criterion(p,y)
            total+=loss.item()*y.size(0); n+=y.size(0)
            Ys.append(y.cpu()); Ps.append(p.cpu())
    Y=torch.cat(Ys,0).numpy(); P=torch.cat(Ps,0).numpy()
    rmse=math.sqrt(((P-Y)**2).mean()); mae=np.abs(P-Y).mean()
    return total/max(1,n), {"rmse":rmse,"mae":mae}, Y, P


def per_horizon(Y,P):
    H=Y.shape[1]
    rmse=[math.sqrt(((P[:,h]-Y[:,h])**2).mean()) for h in range(H)]
    mae=[np.abs(P[:,h]-Y[:,h]).mean() for h in range(H)]
    return rmse, mae

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--code", default="005930")
    ap.add_argument("--device", default="cuda", choices=["cuda","cpu"])
    ap.add_argument("--dump-preds", action="store_true")
    ap.add_argument("--preds-file", default="preds_test.csv")
    args=ap.parse_args()

    device=torch.device(args.device if (args.device=="cpu" or torch.cuda.is_available()) else "cpu")

    payload=load_payload(args.code)
    meta=payload["meta"]; Code=meta["Code"]; horizons=meta["window"]["horizons"]

    ds_va=MIDataset(payload["Xp_va"],payload["Xf_va"],payload["Xd_va"],payload["Y_va"])
    ds_te=MIDataset(payload["Xp_te"],payload["Xf_te"],payload["Xd_te"],payload["Y_te"])

    va_loader=DataLoader(ds_va,batch_size=128,shuffle=False)
    te_loader=DataLoader(ds_te,batch_size=128,shuffle=False)

    # 모델 복원
    P=payload["Xp_tr"].shape[-1]; F=payload["Xf_tr"].shape[-1]; D=payload["Xd_tr"].shape[-1]; H=payload["Y_tr"].shape[-1]
    model=MultiInputLSTM(P,F,D,hidden=96,layers=1,head_hidden=128,out_dim=H,dropout=0.2).to(device)
    
    ckpt_path=Path("models")/f"{Code}_best.pt"
    assert ckpt_path.exists(), f"checkpoint not found: {ckpt_path}"
    
    ckpt=torch.load(ckpt_path,map_location=device)
    model.load_state_dict(ckpt["model"])
    target_kind=ckpt.get("target_kind", meta.get("target_kind","logr"))

    # 평가
    val_loss,val_m, Yv,Pv = evaluate(model,va_loader,device,target_kind)
    te_loss, te_m, Yt,Pt = evaluate(model,te_loader,device,target_kind)

    print(f"[VAL ] loss {val_loss:.4f} | rmse {val_m['rmse']:.4f} | mae {val_m['mae']:.4f}")
    print(f"[TEST] loss {te_loss:.4f} | rmse {te_m['rmse']:.4f} | mae {te_m['mae']:.4f}")

    rmse_v,mae_v = per_horizon(Yv,Pv)
    rmse_t,mae_t = per_horizon(Yt,Pt)
    print("Val  per-horizon RMSE:", [round(x,4) for x in rmse_v])
    print("Val  per-horizon MAE :", [round(x,4) for x in mae_v])
    print("Test per-horizon RMSE:", [round(x,4) for x in rmse_t])
    print("Test per-horizon MAE :", [round(x,4) for x in mae_t])

    if args.dump_preds:
        cols=[f"h{h}" for h in horizons]
        df=pd.DataFrame(np.hstack([Yt,Pt]),columns=[f"y_{c}" for c in cols]+[f"pred_{c}" for c in cols])
        df.to_csv(args.preds_file,index=False)
        print(f"Saved -> {args.preds_file}")

if __name__=="__main__":
    main()
