from __future__ import annotations

import numpy as np


def condition_number(B):
    s=np.linalg.svd(np.asarray(B,float),compute_uv=False)
    if len(s)==0 or s[-1]<1e-15: return float('inf'),s
    return float(s[0]/s[-1]),s


def normalized_coupling(B):
    B=np.asarray(B,float); out=np.full_like(B,np.nan)
    for i in range(B.shape[0]):
        if abs(B[i,i])>1e-12: out[i]=np.abs(B[i])/abs(B[i,i])
    return out


def tikhonov_inverse(B, relative_lambda):
    U,s,Vt=np.linalg.svd(B,full_matrices=False)
    lam=float(relative_lambda)*s[0]**2 if len(s) else 0.0
    gains=np.zeros_like(s)
    if len(s):
        denom=s*s+lam
        nz=denom>1e-15
        gains[nz]=s[nz]/denom[nz]
    D=Vt.T@np.diag(gains)@U.T if len(s) else np.zeros_like(B.T)
    return D,lam,s


def evaluate(B, candidates=None, min_diag=0.70, max_offdiag=0.15, max_gain_ratio=0.60):
    B=np.asarray(B,float)
    pinv=np.linalg.pinv(B)
    pinv_norm=float(np.linalg.norm(pinv,2))
    if candidates is None:
        candidates=[0.0,1e-6,3e-6,1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1,3e-1]
    out=[]
    for rel in candidates:
        D,lam,s=tikhonov_inverse(B,rel)
        BD=B@D; off=BD-np.diag(np.diag(BD))
        dnorm=float(np.linalg.norm(D,2)); ratio=dnorm/max(pinv_norm,1e-15)
        rec={
            'relative_lambda':float(rel),'lambda':float(lam),'D':D,
            'BD':BD,'condition_number':condition_number(B)[0],
            'min_diag':float(np.min(np.diag(BD))),
            'max_abs_offdiag':float(np.max(np.abs(off))),
            'D_norm_2':dnorm,'gain_ratio_vs_pinv':ratio,
            'feasible':bool(rel>0 and np.min(np.diag(BD))>=min_diag and np.max(np.abs(off))<=max_offdiag and ratio<=max_gain_ratio),
            'singular_values':s,
        }
        out.append(rec)
    feasible=[x for x in out if x['feasible']]
    selected=min(feasible,key=lambda x:x['relative_lambda']) if feasible else max(out,key=lambda x:(x['min_diag'],-x['max_abs_offdiag']))
    return selected,out,bool(feasible)
