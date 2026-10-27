"""최종 모델 — Training 전체(판정 구간 앞 1초 이상)로 학습. 속도 측정 스크립트들이 공유한다."""
import importlib
from types import SimpleNamespace

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from sisfall import basic_features, event_window, list_files, shape_features

cnn_mod = importlib.import_module("04_cnn")

SIZE = ["acc_max", "acc_min", "acc_std", "gyro_max", "ff_min"]
SHAPE = ["peak_ratio", "pre_acc_std", "tilt_change", "rot_net", "rot_abs", "vel_down"]
FEATS = SIZE + SHAPE


def features(c: np.ndarray) -> np.ndarray:
    """원신호 2초 창 (401, 6) → 피처 11개 (numpy 기준 구현)."""
    d = basic_features(event_window(c)) | shape_features(c)
    return np.array([d[f] for f in FEATS])


def fit_final_models() -> SimpleNamespace:
    tr = pd.read_pickle("cv_scores_v2.pkl")
    Z, y = tr[FEATS].to_numpy(), tr.is_fall.to_numpy()
    adl = Z[y == 0]
    m = SimpleNamespace()
    m.sc_lr = StandardScaler().fit(Z)
    m.lr = LogisticRegression(max_iter=2000).fit(m.sc_lr.transform(Z), y)
    m.lgbm = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, random_state=0,
                                verbose=-1, n_jobs=1).fit(Z, y)
    m.booster = m.lgbm.booster_
    m.iforest = IsolationForest(n_estimators=300, random_state=0, n_jobs=1).fit(adl)
    m.sc_ae = StandardScaler().fit(adl)
    m.ae = MLPRegressor(hidden_layer_sizes=(8, 3, 8), max_iter=3000, random_state=0).fit(
        m.sc_ae.transform(adl), m.sc_ae.transform(adl))
    m.cnn = cnn_mod.SmallCNN()
    m.cnn.load_state_dict(torch.load("cnn_final.pt"))
    m.cnn.eval()
    return m


def bench_windows(n: int = 2000, seed: int = 0) -> np.ndarray:
    """측정용 입력: Training 원신호 2초 창 n개."""
    w = np.load("windows_v2.npz")
    is_tr = list_files().split.to_numpy()[w["idx"]] == "Training"
    return w["X"][is_tr][np.random.default_rng(seed).choice(is_tr.sum(), n, replace=False)]
