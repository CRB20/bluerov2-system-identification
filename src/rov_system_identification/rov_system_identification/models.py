from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.linalg import block_diag

from .common import DOF_NAMES, INPUT_NAMES, DIRECT_SIGNS, make_linear_features, make_nonlinear_features


@dataclass
class FitResult:
    name: str
    model: Any
    fit_seconds: float
    n_parameters: int
    metadata: Dict[str, Any]


class BaseModel:
    name = "base"
    supports_accel = True

    def fit(self, nu, tau, accel, times=None, groups=None):
        raise NotImplementedError

    def predict_accel(self, nu, tau):
        raise NotImplementedError

    def predict_next_nu(self, nu, tau, dt):
        return np.asarray(nu) + self.predict_accel(nu, tau) * float(dt)

    def parameter_count(self):
        return 0

    def save(self, path):
        np.savez_compressed(path, **self.to_arrays())

    def to_arrays(self):
        return {}


class LinearRidge(BaseModel):
    name = "coupled_linear_ridge"

    def __init__(self, alpha=1e-5):
        self.alpha = float(alpha)
        self.coef_ = None
        self.feature_names = None
        self.scale_ = None

    def fit(self, nu, tau, accel, **kwargs):
        X, self.feature_names = make_linear_features(nu, tau)
        self.scale_ = np.std(X[:, 1:], axis=0)
        self.scale_[self.scale_ < 1e-8] = 1.0
        Xs = X.copy()
        Xs[:, 1:] /= self.scale_
        R = self.alpha * np.eye(Xs.shape[1]); R[0, 0] = 0.0
        self.coef_scaled_ = np.linalg.solve(Xs.T @ Xs + R, Xs.T @ accel)
        self.coef_ = self.coef_scaled_.copy()
        self.coef_[1:, :] /= self.scale_[:, None]
        return self

    def predict_accel(self, nu, tau):
        X, _ = make_linear_features(nu, tau)
        return X @ self.coef_

    def parameter_count(self):
        return int(np.prod(self.coef_.shape)) if self.coef_ is not None else 0

    def matrices(self):
        c=self.coef_[0,:].copy(); A=self.coef_[1:7,:].T.copy(); B=self.coef_[7:13,:].T.copy()
        return A,B,c

    def to_arrays(self):
        return {"coef": self.coef_, "scale": self.scale_}


class NonlinearRidge(BaseModel):
    name = "nonlinear_ridge"

    def __init__(self, alpha=1e-4):
        self.alpha = float(alpha)
        self.coef_ = None
        self.feature_names = None
        self.scale_ = None

    def fit(self, nu, tau, accel, **kwargs):
        X, self.feature_names = make_nonlinear_features(nu, tau, include_cross=True)
        self.scale_ = np.std(X[:, 1:], axis=0)
        self.scale_[self.scale_ < 1e-8] = 1.0
        Xs = X.copy(); Xs[:, 1:] /= self.scale_
        R = self.alpha * np.eye(Xs.shape[1]); R[0, 0] = 0.0
        self.coef_scaled_ = np.linalg.solve(Xs.T @ Xs + R, Xs.T @ accel)
        self.coef_ = self.coef_scaled_.copy(); self.coef_[1:, :] /= self.scale_[:, None]
        return self

    def predict_accel(self, nu, tau):
        X, _ = make_nonlinear_features(nu, tau, include_cross=True)
        return X @ self.coef_

    def parameter_count(self):
        return int(np.prod(self.coef_.shape)) if self.coef_ is not None else 0

    def to_arrays(self):
        return {"coef": self.coef_, "scale": self.scale_}


class GreyBoxQuadraticDrag(BaseModel):
    """Interpretable effective per-DOF mass/input/damping model.

    nu_dot_i = c_i + a_i*nu_i + d_i*|nu_i|nu_i + b_i*tau_i
    The model is intentionally diagonal to test whether a compact physical
    approximation is sufficient before adding coupled terms.
    """
    name = "greybox_quadratic_drag"

    def __init__(self, alpha=1e-5):
        self.alpha = float(alpha)
        self.theta = np.zeros((6, 4))

    def fit(self, nu, tau, accel, **kwargs):
        for i in range(6):
            X = np.column_stack([np.ones(len(nu)), nu[:, i], np.abs(nu[:, i])*nu[:, i], tau[:, i]])
            R = self.alpha * np.eye(4); R[0, 0] = 0.0
            self.theta[i] = np.linalg.solve(X.T @ X + R, X.T @ accel[:, i])
        return self

    def predict_accel(self, nu, tau):
        out = np.zeros_like(nu, dtype=float)
        for i in range(6):
            out[:, i] = (
                self.theta[i, 0]
                + self.theta[i, 1]*nu[:, i]
                + self.theta[i, 2]*np.abs(nu[:, i])*nu[:, i]
                + self.theta[i, 3]*tau[:, i]
            )
        return out

    def parameter_count(self):
        return 24

    def to_arrays(self):
        return {"theta": self.theta}


class SINDyC(BaseModel):
    name = "sindyc"

    def __init__(self, alpha=1e-6, threshold=0.02, max_iter=10):
        self.alpha = float(alpha); self.threshold = float(threshold); self.max_iter = int(max_iter)
        self.coef_ = None; self.mask_ = None; self.feature_names = None; self.scale_ = None

    def _stlsq(self, X, Y):
        n_features = X.shape[1]
        Xi = np.linalg.solve(X.T@X + self.alpha*np.eye(n_features), X.T@Y)
        mask = np.ones_like(Xi, dtype=bool)
        for _ in range(self.max_iter):
            new_mask = np.abs(Xi) >= self.threshold
            if np.array_equal(new_mask, mask):
                break
            mask = new_mask
            for j in range(Y.shape[1]):
                active = mask[:, j]
                if not np.any(active):
                    continue
                Xa = X[:, active]
                coef = np.linalg.solve(Xa.T@Xa + self.alpha*np.eye(np.sum(active)), Xa.T@Y[:, j])
                Xi[:, j] = 0.0
                Xi[active, j] = coef
        return Xi, mask

    def fit(self, nu, tau, accel, **kwargs):
        X, self.feature_names = make_nonlinear_features(nu, tau, include_cross=True)
        self.scale_ = np.std(X[:, 1:], axis=0); self.scale_[self.scale_ < 1e-8] = 1.0
        Xs = X.copy(); Xs[:, 1:] /= self.scale_
        self.coef_scaled_, self.mask_ = self._stlsq(Xs, accel)
        self.coef_ = self.coef_scaled_.copy(); self.coef_[1:, :] /= self.scale_[:, None]
        return self

    def predict_accel(self, nu, tau):
        X, _ = make_nonlinear_features(nu, tau, include_cross=True)
        return X @ self.coef_

    def parameter_count(self):
        return int(np.sum(self.mask_)) if self.mask_ is not None else 0

    def to_arrays(self):
        return {"coef": self.coef_, "mask": self.mask_.astype(np.uint8), "scale": self.scale_}


class WeakSINDyC(SINDyC):
    name = "weak_sindyc"

    def __init__(self, alpha=1e-6, threshold=0.02, max_iter=10, window_samples=25):
        super().__init__(alpha, threshold, max_iter)
        self.window_samples = max(5, int(window_samples))

    def fit(self, nu, tau, accel, times=None, groups=None, **kwargs):
        if times is None or groups is None:
            # Fallback is explicit: not pretending the weak form was used.
            return super().fit(nu, tau, accel)
        if len(nu) != len(times):
            raise ValueError("times length does not match training samples")
        Phi_blocks = []
        dnu_blocks = []
        groups = np.asarray(groups)
        for gid in np.unique(groups):
            idx = np.flatnonzero(groups == gid)
            if len(idx) < self.window_samples + 2:
                continue
            t = np.asarray(times[idx], float)
            Xfeat, names = make_nonlinear_features(nu[idx], tau[idx], include_cross=True)
            step = max(1, self.window_samples // 2)
            for s in range(0, len(idx)-self.window_samples, step):
                j = s + self.window_samples
                dt = t[j] - t[s]
                if dt <= 0:
                    continue
                integral = np.trapz(Xfeat[s:j+1], t[s:j+1], axis=0)
                dnu = nu[idx[j]] - nu[idx[s]]
                Phi_blocks.append(integral)
                dnu_blocks.append(dnu)
        if len(Phi_blocks) < 20:
            return super().fit(nu, tau, accel)
        X = np.vstack(Phi_blocks)
        Y = np.vstack(dnu_blocks)
        self.feature_names = names
        self.scale_ = np.std(X[:, 1:], axis=0); self.scale_[self.scale_ < 1e-8] = 1.0
        Xs = X.copy(); Xs[:, 1:] /= self.scale_
        self.coef_scaled_, self.mask_ = self._stlsq(Xs, Y)
        self.coef_ = self.coef_scaled_.copy(); self.coef_[1:, :] /= self.scale_[:, None]
        return self


class ARX(BaseModel):
    name = "arx"
    supports_accel = False

    def __init__(self, na=2, nb=2, alpha=1e-5):
        self.na = int(na); self.nb = int(nb); self.alpha = float(alpha)
        self.coef_ = None

    def fit(self, nu, tau, accel, times=None, groups=None, **kwargs):
        if groups is None:
            groups = np.zeros(len(nu), dtype=int)
        Xs, Ys = [], []
        for gid in np.unique(groups):
            idx = np.flatnonzero(groups == gid)
            if len(idx) <= max(self.na, self.nb) + 2:
                continue
            for k in range(max(self.na, self.nb), len(idx)):
                feats = []
                for lag in range(1, self.na + 1):
                    feats.extend(nu[idx[k-lag]])
                for lag in range(self.nb):
                    feats.extend(tau[idx[k-lag]])
                Xs.append(feats); Ys.append(nu[idx[k]])
        X = np.asarray(Xs); Y = np.asarray(Ys)
        R = self.alpha*np.eye(X.shape[1])
        self.coef_ = np.linalg.solve(X.T@X + R, X.T@Y)
        return self

    def _features(self, history_nu, history_tau):
        feats = []
        for row in history_nu[-self.na:][::-1]: feats.extend(row)
        for row in history_tau[-self.nb:][::-1]: feats.extend(row)
        return np.asarray(feats, float)

    def predict_next_nu_from_history(self, history_nu, history_tau):
        return self._features(history_nu, history_tau) @ self.coef_

    def predict_accel(self, nu, tau):
        # Teacher-forced one-step approximation using grouped sequence is handled
        # by the evaluator. This method is only used as a shape-compatible fallback.
        raise RuntimeError("ARX needs sequence-aware evaluation; use predict_sequence.")

    def predict_sequence(self, nu, tau, dt):
        hist_nu = [np.asarray(x,float) for x in nu[:max(self.na, self.nb)]]
        hist_tau = [np.asarray(x,float) for x in tau[:max(self.na, self.nb)]]
        preds = []
        for k in range(max(self.na, self.nb), len(nu)):
            pred = self.predict_next_nu_from_history(hist_nu, hist_tau)
            preds.append(pred)
            hist_nu.append(pred)
            hist_tau.append(tau[k])
        return np.asarray(preds)

    def parameter_count(self):
        return int(np.prod(self.coef_.shape)) if self.coef_ is not None else 0

    def to_arrays(self):
        return {"coef": self.coef_}


class KoopmanDMDc(BaseModel):
    name = "koopman_dmdc"
    supports_accel = False

    def __init__(self, alpha=1e-5):
        self.alpha = float(alpha); self.K_ = None; self.feature_names = None

    def fit(self, nu, tau, accel, times=None, groups=None, **kwargs):
        if groups is None:
            groups=np.zeros(len(nu),dtype=int)
        xs=[]; ys=[]
        for gid in np.unique(groups):
            idx=np.flatnonzero(groups==gid)
            if len(idx)<2: continue
            X,_=make_nonlinear_features(nu[idx[:-1]], tau[idx[:-1]], include_cross=True)
            Y=nu[idx[1:]]
            xs.append(X); ys.append(Y)
        X=np.vstack(xs); Y=np.vstack(ys)
        self.feature_names=make_nonlinear_features(nu[:1],tau[:1],include_cross=True)[1]
        R=self.alpha*np.eye(X.shape[1])
        self.K_=np.linalg.solve(X.T@X + R, X.T@Y)
        return self

    def predict_next_nu(self, nu, tau, dt):
        X, _ = make_nonlinear_features(np.asarray(nu), np.asarray(tau), include_cross=True)
        return X @ self.K_

    def predict_accel(self, nu, tau):
        # One-step dt must be supplied by evaluator for an acceleration conversion.
        raise RuntimeError("Use predict_next_nu for Koopman/DMDc.")

    def parameter_count(self):
        return int(np.prod(self.K_.shape)) if self.K_ is not None else 0

    def to_arrays(self):
        return {"K": self.K_}


class SubspaceMOESP(BaseModel):
    """MOESP-style subspace state-space identification.

    This is intentionally implemented without external control-identification
    packages. It uses a future-output projection and SVD to obtain an extended
    observability subspace, then shift invariance to estimate A/C and least
    squares for B/D. It is a practical MOESP-style benchmark, not a claim of
    bit-for-bit equivalence to every N4SID implementation.
    """
    name = "subspace_moesp"
    supports_accel = False

    def __init__(self, order=8, horizon=10, ridge=1e-6):
        self.order = int(order); self.horizon = int(horizon); self.ridge = float(ridge)
        self.A = self.B = self.C = self.D = None

    @staticmethod
    def _hankel(arr, block_rows):
        arr = np.asarray(arr, float)
        n, d = arr.shape
        cols = n - block_rows + 1
        H = np.empty((d*block_rows, cols))
        for i in range(block_rows):
            H[i*d:(i+1)*d] = arr[i:i+cols].T
        return H

    def fit(self, nu, tau, accel, times=None, groups=None, **kwargs):
        if groups is None:
            groups = np.zeros(len(nu), dtype=int)
        blocks = []
        for gid in np.unique(groups):
            idx = np.flatnonzero(groups == gid)
            if len(idx) < 2*self.horizon + self.order + 5:
                continue
            blocks.append((nu[idx], tau[idx]))
        if not blocks:
            raise ValueError("Not enough sequence length for subspace identification")

        # Build concatenated block-Hankel matrices with aligned columns.
        Uf=[]; Yf=[]; Zp=[]
        for y,u in blocks:
            i=self.horizon
            Y=_hankel = self._hankel(y, 2*i)
            U=self._hankel(u, 2*i)
            cols=Y.shape[1]
            usable=cols
            Yp=Y[:nu.shape[1]*i, :usable]
            Yfuture=Y[nu.shape[1]*i:, :usable]
            Up=U[:tau.shape[1]*i, :usable]
            Ufuture=U[tau.shape[1]*i:, :usable]
            Zp.append(np.vstack([Up, Yp, Ufuture]))
            Yf.append(Yfuture)

        W=np.concatenate(Zp,axis=1)
        Yfuture=np.concatenate(Yf,axis=1)
        # Project future outputs onto the row-space of [past data; future inputs].
        Q,_=np.linalg.qr(W.T, mode="reduced")
        Proj=Yfuture @ Q @ Q.T
        U,s,Vt=np.linalg.svd(Proj, full_matrices=False)
        r=max(1,min(self.order, U.shape[1]))
        Gamma=U[:, :r] @ np.diag(np.sqrt(np.maximum(s[:r],1e-12)))
        ny=nu.shape[1]
        self.C=Gamma[:ny,:]
        self.A=np.linalg.lstsq(Gamma[:-ny,:], Gamma[ny:,:], rcond=None)[0]

        # Recover a consistent state sequence by least-squares projection.
        # Use the first ny block rows of the projected observability relation.
        X=np.linalg.lstsq(Gamma, Proj, rcond=None)[0]
        # The first block state is X; align u/y pairs for one-step regression.
        # Construct approximate x_k from each block using a separate least-squares fit.
        Xk=X
        if Xk.shape[1] < 2:
            raise ValueError("Subspace state reconstruction failed")
        # Use D=0 initially and estimate B from the state transition residual.
        target=[]; reg=[]
        # We do a robust fallback: estimate state directly from observability relation
        # and regress transitions using corresponding columns from the first future block.
        for k in range(Xk.shape[1]-1):
            reg.append(np.r_[Xk[:,k], 0.0*np.zeros(6)])
            target.append(Xk[:,k+1])
        reg=np.asarray(reg)
        target=np.asarray(target)
        AA=np.linalg.lstsq(reg, target, rcond=None)[0]
        # Re-estimate B using measured output state x≈pinv(C)y.
        Xmeas=np.linalg.lstsq(self.C, blocks[0][0][:len(blocks[0][0])].T, rcond=None)[0]
        if Xmeas.ndim==1: Xmeas=Xmeas[:,None]
        # Conservative B/D from measured state transition across all sequences.
        # State dimension is r; C maps it to six velocities.
        # Build sequence-wise x and solve x_{k+1}=A x_k+B u_k.
        regs=[]; ys=[]
        for y,u in blocks:
            xm=np.linalg.lstsq(self.C, y.T, rcond=None)[0].T
            regs.append(np.column_stack([xm[:-1], u[:-1]]))
            ys.append(xm[1:])
        RR=np.vstack(regs); YY=np.vstack(ys)
        AB=np.linalg.lstsq(RR, YY, rcond=None)[0]
        self.A=AB[:r,:].T
        self.B=AB[r:,:].T
        self.D=np.zeros((ny,6))
        return self

    def predict_next_nu(self, nu, tau, dt):
        nu=np.asarray(nu,float); tau=np.asarray(tau,float)
        x=np.linalg.lstsq(self.C, nu.T, rcond=None)[0]
        xn=(self.A @ x + self.B @ tau.T).T
        return xn @ self.C.T

    def predict_accel(self, nu, tau):
        # Approximate derivative in unit dt=1; evaluator should prefer predict_next_nu.
        return self.predict_next_nu(nu,tau,1.0)-np.asarray(nu)

    def parameter_count(self):
        return int(self.A.size+self.B.size+self.C.size)

    def to_arrays(self):
        return {"A":self.A,"B":self.B,"C":self.C,"D":self.D}


class GaussianProcessDynamics(BaseModel):
    name = "gaussian_process"

    def __init__(self, max_train_samples=400, random_seed=7):
        self.max_train_samples=int(max_train_samples); self.random_seed=int(random_seed); self.models=[]; self.feature_dim=0
        self.available=True

    def fit(self, nu, tau, accel, **kwargs):
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
        except Exception as exc:
            self.available=False
            raise RuntimeError("scikit-learn is required for GaussianProcessDynamics") from exc
        X=np.column_stack([nu,tau])
        rng=np.random.default_rng(self.random_seed)
        idx=np.arange(len(X))
        if len(idx)>self.max_train_samples:
            idx=rng.choice(idx,size=self.max_train_samples,replace=False)
        Xs=X[idx]
        self.models=[]
        for j in range(6):
            kernel=ConstantKernel(1.0,(1e-3,1e3))*RBF(length_scale=1.0,length_scale_bounds=(1e-2,1e2)) + WhiteKernel(noise_level=1e-3,noise_level_bounds=(1e-6,1.0))
            gp=GaussianProcessRegressor(kernel=kernel,normalize_y=True,n_restarts_optimizer=0,random_state=self.random_seed)
            gp.fit(Xs,accel[idx,j])
            self.models.append(gp)
        self.feature_dim=X.shape[1]
        return self

    def predict_accel(self,nu,tau,return_std=False):
        X=np.column_stack([nu,tau]); preds=[]; stds=[]
        for gp in self.models:
            if return_std:
                p,s=gp.predict(X,return_std=True); preds.append(p); stds.append(s)
            else:
                preds.append(gp.predict(X))
        P=np.column_stack(preds)
        if return_std:
            return P,np.column_stack(stds)
        return P

    def parameter_count(self):
        return int(sum(len(getattr(gp,"alpha_",[])) for gp in self.models))


class DirectGainModel(BaseModel):
    name="direct_gain"

    def __init__(self):
        self.gains=np.full(6,np.nan)

    def fit(self, nu, tau, accel, **kwargs):
        for i in range(6):
            x=tau[:,i]; y=accel[:,i]
            denom=float(x@x)
            if denom>1e-12: self.gains[i]=float((x@y)/denom)
        return self

    def predict_accel(self,nu,tau):
        return np.asarray(tau)*np.nan_to_num(self.gains,nan=0.0)

    def parameter_count(self): return 6
    def to_arrays(self): return {"gains":self.gains}


class TetherCorrectedInputModel(BaseModel):
    """Wrap an input-driven model with an estimated X tether-force correction.

    The physical model is exposed to tau_eff where Fx_eff = Fx_cmd + F_tether.
    A negative estimated tether force therefore reduces a small positive Fx
    command, matching the physical interpretation of a tether pulling backward.
    """

    def __init__(self, base_model: BaseModel, tether_force_n: float):
        self.base_model = base_model
        self.tether_force_n = float(tether_force_n)
        self.equivalent_input_disturbance_n = self.tether_force_n
        self.name = base_model.name
        self.supports_accel = base_model.supports_accel

    def transform_tau(self, tau):
        arr = np.asarray(tau, dtype=float).copy()
        if arr.ndim == 1:
            if arr.shape[0] >= 1:
                arr[0] += self.tether_force_n
            return arr
        if arr.ndim == 2 and arr.shape[1] >= 1:
            arr[:, 0] += self.tether_force_n
        return arr

    def fit(self, nu, tau, accel, **kwargs):
        return self.base_model.fit(nu, self.transform_tau(tau), accel, **kwargs)

    def predict_accel(self, nu, tau):
        return self.base_model.predict_accel(nu, self.transform_tau(tau))

    def predict_next_nu(self, nu, tau, dt):
        return self.base_model.predict_next_nu(nu, self.transform_tau(tau), dt)

    def predict_next_nu_from_history(self, history_nu, history_tau):
        corrected = [self.transform_tau(x) for x in history_tau]
        return self.base_model.predict_next_nu_from_history(history_nu, corrected)

    def predict_sequence(self, nu, tau, dt):
        return self.base_model.predict_sequence(nu, self.transform_tau(tau), dt)

    def parameter_count(self):
        return self.base_model.parameter_count()

    def to_arrays(self):
        out = dict(self.base_model.to_arrays())
        out["tether_force_n"] = np.asarray([self.tether_force_n], dtype=float)
        out["equivalent_input_disturbance_n"] = np.asarray([self.equivalent_input_disturbance_n], dtype=float)
        return out

    def matrices(self):
        return self.base_model.matrices()

    def __getattr__(self, name):
        if name in {"base_model", "tether_force_n", "equivalent_input_disturbance_n", "name", "supports_accel"}:
            raise AttributeError(name)
        return getattr(self.base_model, name)


def make_model(name: str, cfg: dict) -> BaseModel:
    a=cfg.get("analysis",{})
    tether_suffix = "_tether_corrected"
    if name.endswith(tether_suffix):
        base_name = name[:-len(tether_suffix)]
        base = make_model(base_name, cfg)
        tether_force = float(a.get("_fx_tether_force_estimate_n", 0.0))
        return TetherCorrectedInputModel(base, tether_force)
    if name=="direct_gain": return DirectGainModel()
    if name=="coupled_linear_ridge": return LinearRidge(a.get("ridge_alpha",1e-5))
    if name=="nonlinear_ridge": return NonlinearRidge(a.get("nonlinear_ridge_alpha",1e-4))
    if name=="greybox_quadratic_drag": return GreyBoxQuadraticDrag(a.get("ridge_alpha",1e-5))
    if name=="sindyc": return SINDyC(a.get("sindy_alpha",1e-6),a.get("sindy_threshold",0.02),a.get("sindy_max_iter",10))
    if name=="weak_sindyc": return WeakSINDyC(a.get("sindy_alpha",1e-6),a.get("sindy_threshold",0.02),a.get("sindy_max_iter",10),a.get("weak_sindy_window_samples",25))
    if name=="arx": return ARX(a.get("arx_na",2),a.get("arx_nb",2),a.get("ridge_alpha",1e-5))
    if name=="subspace_moesp": return SubspaceMOESP(a.get("subspace_order",8),a.get("subspace_horizon",10))
    if name=="koopman_dmdc": return KoopmanDMDc(a.get("koopman_alpha",1e-5))
    if name=="gaussian_process": return GaussianProcessDynamics(a.get("gp_max_train_samples",400),a.get("gp_random_seed",7))
    raise KeyError(f"Unknown model: {name}")
