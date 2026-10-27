"""4단계 — 판정 1회당 추론 비용. 원신호 2초가 들어와서 점수가 나올 때까지(피처 계산 포함).

- 노트북 CPU 1스레드에서 측정
  · 한 건씩: 실시간처럼 창 하나를 넣고 점수 하나를 받는다. 파이썬 라이브러리 호출 비용이 포함된다
  · 묶음 평균: 2,000건을 한 번에 넣고 건수로 나눈다. 호출 비용이 희석되어 순수 계산량에 가깝다
- 칩과 무관한 지표: 판정 1회당 연산 수, 모델 크기(float32 기준)
"""
import os

for v in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ[v] = "1"

import time

import numpy as np
import pandas as pd
import torch

from final_models import FEATS, bench_windows, cnn_mod, features, fit_final_models
from sisfall import EVENT_LEN

torch.set_num_threads(1)
pd.set_option("display.width", 200)

m = fit_final_models()
sc_lr, lr, booster, iforest, sc_ae, ae, cnn = m.sc_lr, m.lr, m.booster, m.iforest, m.sc_ae, m.ae, m.cnn
W = bench_windows()


def run_baseline(c):
    return np.linalg.norm(c[-EVENT_LEN:, :3], axis=1).max()


def run_lr(c):
    return lr.predict_proba(sc_lr.transform(features(c)[None]))[0, 1]


def run_lgbm(c):
    return booster.predict(features(c)[None], num_threads=1)[0]


def run_iforest(c):
    return -iforest.score_samples(features(c)[None])[0]


def run_ae(c):
    z = sc_ae.transform(features(c)[None])
    return ((ae.predict(z) - z) ** 2).mean()


def run_cnn(c):
    with torch.inference_mode():
        return torch.sigmoid(cnn(torch.from_numpy(cnn_mod.to_input(c)[None])))[0].item()


def per_call_us(fn):
    for c in W[:100]:
        fn(c)  # 예열
    ts = []
    for c in W:
        t0 = time.perf_counter_ns()
        fn(c)
        ts.append(time.perf_counter_ns() - t0)
    return np.median(ts) / 1e3, np.percentile(ts, 99) / 1e3


def batch_us(fn_batch, X):
    fn_batch(X[:100])
    t0 = time.perf_counter_ns()
    fn_batch(X)
    return (time.perf_counter_ns() - t0) / 1e3 / len(X)


ZW = np.stack([features(c) for c in W])
CW = np.stack([cnn_mod.to_input(c) for c in W])


def cnn_batch(X):
    with torch.inference_mode():
        return cnn(torch.from_numpy(X))


rows = {
    "피처 계산만": (per_call_us(features), None),
    "기준선: 최고 가속도": (per_call_us(run_baseline), batch_us(lambda X: np.linalg.norm(X[:, -EVENT_LEN:, :3], axis=2).max(1), W)),
    "로지스틱 회귀": (per_call_us(run_lr), batch_us(lambda X: lr.predict_proba(sc_lr.transform(X)), ZW)),
    "LightGBM": (per_call_us(run_lgbm), batch_us(lambda X: booster.predict(X, num_threads=1), ZW)),
    "Isolation Forest": (per_call_us(run_iforest), batch_us(iforest.score_samples, ZW)),
    "오토인코더": (per_call_us(run_ae), batch_us(lambda X: ae.predict(sc_ae.transform(X)), ZW)),
    "1D CNN": (per_call_us(run_cnn), batch_us(cnn_batch, CW)),
}

# ── 칩과 무관한 지표
FEAT_OPS = 10_000  # 피처 11개 계산 ≈ 2초(401샘플×6채널)에 대한 노름·최대·표준편차·누적합 → 약 1만 회 (추정)


def lgbm_cost():
    dump = booster.dump_model()["tree_info"]
    depth, n_nodes = [], 0
    for t in dump:
        d = {}

        def walk(node, k):
            nonlocal n_nodes
            if "leaf_index" in node:
                d[node["leaf_index"]] = k
                n_nodes += 1
            else:
                n_nodes += 2  # 분기 노드: 피처 번호 + 임계값 (잎은 값 1개)
                walk(node["left_child"], k + 1)
                walk(node["right_child"], k + 1)

        walk(t["tree_structure"], 0)
        depth.append(d)
    leaves = booster.predict(ZW, pred_leaf=True)
    cmp = np.mean([sum(depth[j][leaves[i, j]] for j in range(len(depth))) for i in range(len(ZW))])
    return cmp, n_nodes


def iforest_cost():
    cmp, n = 0.0, 0
    for est, f in zip(iforest.estimators_, iforest.estimators_features_):
        cmp += (np.asarray(est.decision_path(ZW[:, f]).sum(1)).ravel() - 1).mean()
        t = est.tree_
        n += 2 * (t.children_left != -1).sum() + (t.children_left == -1).sum()
    return cmp, n


def cnn_macs():
    macs = []

    def hook(m, i, o):
        if isinstance(m, torch.nn.Conv1d):
            macs.append(o.shape[-1] * m.out_channels * m.in_channels * m.kernel_size[0])
        elif isinstance(m, torch.nn.Linear):
            macs.append(m.in_features * m.out_features)

    hs = [m.register_forward_hook(hook) for m in cnn.modules() if isinstance(m, (torch.nn.Conv1d, torch.nn.Linear))]
    cnn_batch(CW[:1])
    [h.remove() for h in hs]
    return sum(macs), sum(p.numel() for p in cnn.parameters())


lg_cmp, lg_n = lgbm_cost()
if_cmp, if_n = iforest_cost()
cnn_ops, cnn_n = cnn_macs()
ae_macs = sum(a.size for a in ae.coefs_)
ae_n = ae_macs + sum(b.size for b in ae.intercepts_) + 2 * len(FEATS)
ops = {  # (판정 1회 연산 수, 저장할 숫자 개수)
    "기준선: 최고 가속도": (EVENT_LEN * 7, 1),
    "로지스틱 회귀": (FEAT_OPS + len(FEATS), len(FEATS) + 1),  # 표준화는 가중치에 합칠 수 있다
    "LightGBM": (FEAT_OPS + lg_cmp, lg_n),
    "Isolation Forest": (FEAT_OPS + if_cmp, if_n),
    "오토인코더": (FEAT_OPS + ae_macs, ae_n),
    "1D CNN": (cnn_ops, cnn_n),
}

out = pd.DataFrame({
    "한 건씩 중앙값(µs)": {k: v[0][0] for k, v in rows.items()},
    "한 건씩 99%(µs)": {k: v[0][1] for k, v in rows.items()},
    "묶음 평균(µs/건)": {k: v[1] for k, v in rows.items()},
    "연산 수/판정": {k: v[0] for k, v in ops.items()},
    "모델 크기(KB)": {k: v[1] * 4 / 1024 for k, v in ops.items()},
})
print(out.round(2).to_string())
print(f"\n세부: LightGBM 평균 비교 {lg_cmp:.0f}회, Isolation Forest 평균 비교 {if_cmp:.0f}회, "
      f"CNN 곱셈-덧셈 {cnn_ops:,}회, CNN 파라미터 {cnn_n:,}개")
out.to_pickle("speed_v1.pkl")
