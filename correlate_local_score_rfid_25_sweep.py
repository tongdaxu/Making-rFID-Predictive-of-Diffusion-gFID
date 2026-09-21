#!/usr/bin/env python3
import argparse, re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau

def canon(s):
    s = str(s).lower()
    aliases = {
        "qwenimgvae": "qwvae", "qwenimagevae": "qwvae",
        "repaesdvae": "repaevae", "repae-sdvae": "repaevae",
        "flux2-vae": "flux2vae", "sd-vae": "sdvae", "sd3-vae": "sd3vae",
    }
    for a,b in aliases.items(): s=s.replace(a,b)
    return re.sub(r"[^a-z0-9]+", "", s)

def corr(x,y):
    m=np.isfinite(x)&np.isfinite(y); x=x[m]; y=y[m]
    if len(x)<3: return len(x),np.nan,np.nan,np.nan
    return len(x),pearsonr(x,y).statistic,spearmanr(x,y).statistic,kendalltau(x,y).statistic

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--ls-csv", required=True)
    ap.add_argument("--gfid-csv", required=True)
    ap.add_argument("--model-col", default="model")
    ap.add_argument("--gfid-cols", nargs="*", default=None)
    ap.add_argument("--out", default=None)
    a=ap.parse_args()
    ls=pd.read_csv(a.ls_csv); gf=pd.read_csv(a.gfid_csv)
    ls=ls[ls.status.fillna("")=="ok"].copy()
    if a.model_col not in gf.columns: raise KeyError(f"missing {a.model_col}; {list(gf.columns)}")
    ls["_k"]=ls.model.map(canon); gf["_k"]=gf[a.model_col].map(canon)
    m=ls.merge(gf,on="_k",suffixes=("_ls","_gfid"))
    metric_cols=[c for c in ls.columns if c=="rfid" or c.startswith("ls_rfid_")]
    if a.gfid_cols:
        gcols=a.gfid_cols
    else:
        gcols=[]
        for c in gf.columns:
            if c in (a.model_col,"_k"): continue
            if pd.to_numeric(m[c], errors="coerce").notna().sum()>=3: gcols.append(c)
    rows=[]
    for mc in metric_cols:
        x=pd.to_numeric(m[mc],errors="coerce").to_numpy(float)
        for gc in gcols:
            y=pd.to_numeric(m[gc],errors="coerce").to_numpy(float)
            n,p,s,k=corr(x,y)
            rows.append(dict(metric=mc,gfid_column=gc,n=n,PCC=p,SRCC=s,KRCC=k))
    out=Path(a.out) if a.out else Path(a.ls_csv).parent/"local_score_p_sweep_correlations.csv"
    pd.DataFrame(rows).to_csv(out,index=False)
    print(f"matched models: {len(m)}")
    print(out)
    print(pd.DataFrame(rows).to_string(index=False))

if __name__=="__main__": main()
