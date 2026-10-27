"""4단계 — 배포에 가까운 조건에서 판정 1회당 지연 시간.

05의 노트북 측정은 파이썬 라이브러리 호출 비용이 대부분이라 모델 간 속도 순위를 매길 수 없었다.
여기서는 두 방식으로 다시 잰다. 둘 다 CPU 1스레드, 원신호 2초 창을 한 건씩 넣는다(실시간처럼).
  A. 컴파일 구현: 피처 계산과 모델을 numba로 기계어 컴파일 → 기기용 C 코드로 옮겼을 때와 가장 가깝다
  B. ONNX Runtime: 기기 배포에 흔히 쓰는 추론 엔진. 피처 계산은 A의 컴파일 버전을 쓴다
각 구현이 원래 라이브러리와 같은 점수를 내는지 먼저 검증한다.
"""
import os

for v in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ[v] = "1"

import sys
import time
from math import acos, exp, pi, sqrt

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
from numba import njit
from onnxmltools import convert_lightgbm
from onnxmltools.convert.common.data_types import FloatTensorType
from scipy.stats import spearmanr
from skl2onnx import to_onnx
from sklearn.pipeline import make_pipeline

from final_models import FEATS, bench_windows, cnn_mod, features, fit_final_models

torch.set_num_threads(1)
pd.set_option("display.width", 200)
m = fit_final_models()
W = bench_windows()


# ════════════════ A. 컴파일 구현 ════════════════
@njit(cache=True)
def features_nb(c):
    """sisfall.basic_features + shape_features 와 같은 계산. 노름은 한 번만 구해 공유한다."""
    n = c.shape[0]
    P = n - 201  # 판정 구간 시작 = 앞 1초 길이(200)
    acc = np.empty(n)
    gyr = np.empty(n)
    for i in range(n):
        acc[i] = sqrt(c[i, 0] ** 2 + c[i, 1] ** 2 + c[i, 2] ** 2)
        gyr[i] = sqrt(c[i, 3] ** 2 + c[i, 4] ** 2 + c[i, 5] ** 2)
    k, amax, amin, gmax, s = P, acc[P], acc[P], gyr[P], 0.0
    for i in range(P, n):
        if acc[i] > amax:
            amax, k = acc[i], i
        amin = min(amin, acc[i])
        gmax = max(gmax, gyr[i])
        s += acc[i]
    mu, ss = s / (n - P), 0.0
    for i in range(P, n):
        ss += (acc[i] - mu) ** 2
    ff = acc[P]
    for i in range(P, k + 1):
        ff = min(ff, acc[i])
    pmax, s = acc[0], 0.0
    for i in range(P):
        pmax = max(pmax, acc[i])
        s += acc[i]
    pmu, pss = s / P, 0.0
    for i in range(P):
        pss += (acc[i] - pmu) ** 2
    g0 = np.zeros(3)
    g1 = np.zeros(3)
    for i in range(50):
        for j in range(3):
            g0[j] += c[i, j]
    for i in range(n - 20, n):
        for j in range(3):
            g1[j] += c[i, j]
    cos = (g0[0] * g1[0] + g0[1] * g1[1] + g0[2] * g1[2]) / (
        sqrt(g0[0] ** 2 + g0[1] ** 2 + g0[2] ** 2) * sqrt(g1[0] ** 2 + g1[1] ** 2 + g1[2] ** 2))
    wx = wy = wz = rabs = 0.0
    for i in range(n):
        wx += c[i, 3]
        wy += c[i, 4]
        wz += c[i, 5]
        rabs += gyr[i]
    v, vmin = 0.0, 1e300
    for i in range(k - 200, k):
        v += (acc[i] - 1.0) * 9.81 / 200
        vmin = min(vmin, v)
    out = np.empty(11)
    out[0], out[1], out[2], out[3], out[4] = amax, amin, sqrt(ss / (n - P)), gmax, ff
    out[5], out[6] = amax / pmax, sqrt(pss / P)
    out[7] = acos(min(1.0, max(-1.0, cos))) * 180 / pi
    out[8] = sqrt(wx ** 2 + wy ** 2 + wz ** 2) / 200
    out[9], out[10] = rabs / 200, -vmin
    return out


@njit(cache=True)
def baseline_nb(c):
    n, amax = c.shape[0], 0.0
    for i in range(n - 201, n):
        amax = max(amax, sqrt(c[i, 0] ** 2 + c[i, 1] ** 2 + c[i, 2] ** 2))
    return amax


@njit(cache=True)
def lr_nb(c, w, b):
    f = features_nb(c)
    z = b
    for j in range(f.shape[0]):
        z += w[j] * f[j]
    return 1 / (1 + exp(-z))


@njit(cache=True)
def lgbm_nb(c, roots, feat, thr, left, right, leafv):
    f = features_nb(c)
    s = 0.0
    for t in range(roots.shape[0]):
        node = roots[t]
        while node >= 0:
            node = left[node] if f[feat[node]] <= thr[node] else right[node]
        s += leafv[-node - 1]
    return 1 / (1 + exp(-s))


@njit(cache=True)
def dense(x, Wt, b, relu):
    y = b.copy()
    for o in range(Wt.shape[1]):
        for i in range(Wt.shape[0]):
            y[o] += x[i] * Wt[i, o]
        if relu and y[o] < 0:
            y[o] = 0.0
    return y


@njit(cache=True)
def ae_nb(c, mean, scale, W1, b1, W2, b2, W3, b3, W4, b4):
    x = (features_nb(c) - mean) / scale
    r = dense(dense(dense(dense(x, W1, b1, True), W2, b2, True), W3, b3, True), W4, b4, False)
    return ((r - x) ** 2).mean()


@njit(cache=True)
def conv_relu(x, Wt, b, stride):
    co, ci, K = Wt.shape
    L = (x.shape[1] - K) // stride + 1
    y = np.empty((co, L), np.float32)
    for o in range(co):
        for t in range(L):
            s = b[o]
            for i in range(ci):
                for k in range(K):
                    s += Wt[o, i, k] * x[i, t * stride + k]
            y[o, t] = s if s > 0 else np.float32(0)
    return y


@njit(cache=True)
def cnn_nb(c, W1, b1, W2, b2, W3, b3, Wh, bh):
    n = c.shape[0]
    x = np.empty((6, n), np.float32)
    for t in range(n):
        for i in range(6):
            x[i, t] = c[t, i] if i < 3 else c[t, i] / 500.0  # 04_cnn.GYRO_SCALE
    h = conv_relu(conv_relu(conv_relu(x, W1, b1, 2), W2, b2, 2), W3, b3, 2)
    C, L = h.shape
    z = float(bh[0])
    for o in range(C):
        s, mx = 0.0, h[o, 0]
        for t in range(L):
            s += h[o, t]
            mx = max(mx, h[o, t])
        z += Wh[0, o] * (s / L) + Wh[0, C + o] * mx
    return 1 / (1 + exp(-z))


# 모델 파라미터를 배열로 꺼낸다
lr_w = m.lr.coef_[0] / m.sc_lr.scale_  # 표준화를 가중치에 합친다
lr_b = m.lr.intercept_[0] - (m.lr.coef_[0] * m.sc_lr.mean_ / m.sc_lr.scale_).sum()


def flatten_lgbm(booster):
    feat, thr, left, right, leafv, roots = [], [], [], [], [], []

    def add(node):
        if "leaf_index" in node:
            leafv.append(node["leaf_value"])
            return -len(leafv)  # 음수 = 잎 번호
        assert node["decision_type"] == "<=" and node["missing_type"] in ("None", "NaN")
        i = len(feat)
        feat.append(node["split_feature"]); thr.append(node["threshold"]); left.append(0); right.append(0)
        left[i], right[i] = add(node["left_child"]), add(node["right_child"])
        return i

    for t in booster.dump_model()["tree_info"]:
        roots.append(add(t["tree_structure"]))
    return tuple(np.array(a) for a in (roots, feat, thr, left, right, leafv))


lg = flatten_lgbm(m.booster)
ae_p = (m.sc_ae.mean_, m.sc_ae.scale_, *[a for pair in zip(m.ae.coefs_, m.ae.intercepts_) for a in pair])
sd = {k: v.numpy() for k, v in m.cnn.state_dict().items()}
cnn_p = (sd["conv.0.weight"], sd["conv.0.bias"], sd["conv.2.weight"], sd["conv.2.bias"],
         sd["conv.4.weight"], sd["conv.4.bias"], sd["head.weight"], sd["head.bias"])

A = {
    "피처 계산만": lambda c: features_nb(c),
    "기준선: 최고 가속도": lambda c: baseline_nb(c),
    "로지스틱 회귀": lambda c: lr_nb(c, lr_w, lr_b),
    "LightGBM": lambda c: lgbm_nb(c, *lg),
    "오토인코더": lambda c: ae_nb(c, *ae_p),
    "1D CNN": lambda c: cnn_nb(c, *cnn_p),
}

# ── 검증: 원래 라이브러리 점수와 비교
V = W[:500]
ZV = np.stack([features(c) for c in V])
with torch.inference_mode():
    cnn_ref = torch.sigmoid(m.cnn(torch.from_numpy(np.stack([cnn_mod.to_input(c) for c in V])))).numpy()
ref = {
    "피처 계산만": ZV,
    "기준선: 최고 가속도": ZV[:, 0],
    "로지스틱 회귀": m.lr.predict_proba(m.sc_lr.transform(ZV))[:, 1],
    "LightGBM": m.booster.predict(ZV),
    "오토인코더": ((m.ae.predict(m.sc_ae.transform(ZV)) - m.sc_ae.transform(ZV)) ** 2).mean(1),
    "1D CNN": cnn_ref,
}
print("== A. 컴파일 구현 검증: 원래 라이브러리와의 최대 절대 차이 ==")
for name, fn in A.items():
    got = np.stack([fn(c) for c in V])
    print(f"  {name:12s} {np.abs(got - ref[name]).max():.2e}")


def time_per_call(fn, X=W):
    for c in X[:200]:
        fn(c)  # 예열(컴파일 포함)
    ts = np.empty(len(X))
    for j, c in enumerate(X):
        t0 = time.perf_counter_ns()
        fn(c)
        ts[j] = time.perf_counter_ns() - t0
    return np.median(ts) / 1e3, np.percentile(ts, 99) / 1e3


# ════════════════ B. ONNX Runtime ════════════════
so = ort.SessionOptions()
so.intra_op_num_threads = so.inter_op_num_threads = 1
so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
X32 = ZV.astype(np.float32)
onnx_models = {
    "로지스틱 회귀": to_onnx(make_pipeline(m.sc_lr, m.lr), X32[:1], options={id(m.lr): {"zipmap": False}}),
    "LightGBM": convert_lightgbm(m.lgbm, initial_types=[("X", FloatTensorType([None, len(FEATS)]))], zipmap=False),
    "Isolation Forest": to_onnx(m.iforest, X32[:1], target_opset={"": 17, "ai.onnx.ml": 3}),
}  # 오토인코더는 제외: skl2onnx 가 다중 출력 MLPRegressor 를 출력 1개로 잘못 변환한다(검증에서 확인)
torch.onnx.export(m.cnn, torch.zeros(1, 6, 401), "cnn_final.onnx", input_names=["x"], output_names=["y"])
sess = {k: ort.InferenceSession(v.SerializeToString(), so) for k, v in onnx_models.items()}
sess["1D CNN"] = ort.InferenceSession("cnn_final.onnx", so)
inp = {k: s.get_inputs()[0].name for k, s in sess.items()}


def onnx_fn(name):
    s, i = sess[name], inp[name]
    if name == "1D CNN":
        return lambda c: s.run(None, {i: cnn_mod.to_input(c)[None]})
    return lambda c: s.run(None, {i: features_nb(c).astype(np.float32)[None]})


B = {k: onnx_fn(k) for k in sess}

print("\n== B. ONNX Runtime 검증 ==")
pick = {"로지스틱 회귀": lambda o: o[1][:, 1], "LightGBM": lambda o: o[1][:, 1], "Isolation Forest": lambda o: o[1].ravel()}
for name in ["로지스틱 회귀", "LightGBM"]:
    got = pick[name](sess[name].run(None, {inp[name]: X32}))
    print(f"  {name:12s} 최대 절대 차이 {np.abs(got - ref[name]).max():.2e}")
got = pick["Isolation Forest"](sess["Isolation Forest"].run(None, {inp["Isolation Forest"]: X32}))
print(f"  {'Isolation Forest':12s} 원래 점수와 순위 상관 {spearmanr(got, m.iforest.score_samples(ZV)).statistic:.4f}")
got = np.array([sess["1D CNN"].run(None, {"x": cnn_mod.to_input(c)[None]})[0][0] for c in V])
print(f"  {'1D CNN':12s} 최대 절대 차이(로짓→확률) {np.abs(1 / (1 + np.exp(-got)) - ref['1D CNN']).max():.2e}")

# ════════════════ 측정 ════════════════
res = {}
for name, fn in A.items():
    res[(name, "A")] = time_per_call(fn)
for name, fn in B.items():
    res[(name, "B")] = time_per_call(fn)
order = ["피처 계산만", "기준선: 최고 가속도", "로지스틱 회귀", "LightGBM", "Isolation Forest", "오토인코더", "1D CNN"]
tab = pd.DataFrame({
    "A 컴파일 중앙값(µs)": {k: res.get((k, "A"), (np.nan, np.nan))[0] for k in order},
    "A 컴파일 99%(µs)": {k: res.get((k, "A"), (np.nan, np.nan))[1] for k in order},
    "B ONNX 중앙값(µs)": {k: res.get((k, "B"), (np.nan, np.nan))[0] for k in order},
    "B ONNX 99%(µs)": {k: res.get((k, "B"), (np.nan, np.nan))[1] for k in order},
})
print("\n== 판정 1회당 지연 시간 (원신호 2초 → 점수, 1스레드, 한 건씩) ==")
print(tab.round(2).to_string())
tab.to_pickle("speed_v2.pkl")
sys.stdout.flush()
os._exit(0)  # onnxruntime·torch 종료 정리 중 macOS에서 mutex 오류가 나는 것을 피한다(결과 저장 후)
