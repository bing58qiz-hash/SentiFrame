"""Test 평가 — 후보 6종을 한 번만 평가하고, 성능과 추론 속도를 표 하나로 묶는다.

- 모델: Training 전체로 학습한 최종 모델(final_models, cnn_final.pt)
- 임계값: Training 교차검증 점수(cv_scores_v3)에서 재현율 90/95/99%가 되는 값을 정해 그대로 적용한다.
  Test에서 임계값을 다시 맞추지 않는다 — 기기도 정해둔 임계값으로 동작하기 때문
- 불확실성: Test 8명을 사람 단위로 다시 뽑는 부트스트랩(청년·고령자 따로) 95% 구간
"""
import os

for v in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ[v] = "1"  # LightGBM 과 PyTorch 가 각자 OpenMP 를 불러 다중 스레드로 돌면 세그폴트가 난다

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from evaluation import threshold_at_recall
from final_models import FEATS, cnn_mod, fit_final_models
from sisfall import ACTIVITY_NAMES, load_table

pd.set_option("display.width", 250)
torch.set_num_threads(1)

df = load_table()
te_all = df[df.split == "Test"]
print(f"Test: {te_all.subject.nunique()}명, 낙상 {te_all.is_fall.sum()}, 일상동작 {(te_all.is_fall == 0).sum()} "
      f"→ 앞 1초 미만 제외: 낙상 {((te_all.pre_s < 1) & (te_all.is_fall == 1)).sum()}, "
      f"일상동작 {((te_all.pre_s < 1) & (te_all.is_fall == 0)).sum()}")
print("제외된 Test 파일:", te_all[te_all.pre_s < 1][["subject", "activity", "rep", "pre_s"]].to_dict("records"))
keep = (df.split == "Test") & (df.pre_s >= 1)
te = df[keep].reset_index(drop=True)
Z, y = te[FEATS].to_numpy(), te.is_fall.to_numpy()

# ── 최종 모델로 Test 점수
m = fit_final_models()
w = np.load("windows_v2.npz")
pos = {j: i for i, j in enumerate(w["idx"])}
Xc = np.stack([cnn_mod.to_input(w["X"][pos[j]]) for j in np.flatnonzero(keep.to_numpy())])
with torch.inference_mode():
    cnn_score = torch.sigmoid(m.cnn(torch.from_numpy(Xc))).numpy()
zs = m.sc_ae.transform(Z)
scores = {
    "기준선: 최고 가속도": ("s_baseline", te.acc_max.to_numpy()),
    "로지스틱 회귀": ("s_logreg", m.lr.predict_proba(m.sc_lr.transform(Z))[:, 1]),
    "LightGBM": ("s_lgbm", m.booster.predict(Z)),
    "1D CNN": ("s_cnn", cnn_score),
    "Isolation Forest": ("s_iforest", -m.iforest.score_samples(Z)),
    "오토인코더": ("s_ae", ((m.ae.predict(zs) - zs) ** 2).mean(1)),
}

cv = pd.read_pickle("cv_scores_v3.pkl")
rng = np.random.default_rng(0)
young, elder = te[te.group == "SA"].subject.unique(), te[te.group == "SE"].subject.unique()
by_subj = {s: np.flatnonzero(te.subject.to_numpy() == s) for s in te.subject.unique()}


def boot_ci(alarm, n=2000):
    rec, spe = [], []
    for _ in range(n):
        subs = np.r_[rng.choice(young, len(young)), rng.choice(elder, len(elder))]
        idx = np.concatenate([by_subj[s] for s in subs])
        rec.append(alarm[idx][y[idx] == 1].mean())
        spe.append(1 - alarm[idx][y[idx] == 0].mean())
    return np.percentile(rec, [2.5, 97.5]), np.percentile(spe, [2.5, 97.5])


rows, detail = [], {}
for name, (col, s) in scores.items():
    r = dict(후보=name, Test_AUC=roc_auc_score(y, s))
    for target in (0.90, 0.95, 0.99):
        thr = threshold_at_recall(cv.is_fall, cv[col], target)
        alarm = s >= thr
        r[f"재현율@{round(target * 100)}"] = alarm[y == 1].mean()
        r[f"특이도@{round(target * 100)}"] = 1 - alarm[y == 0].mean()
        if target == 0.95:
            (r["재현율95_lo"], r["재현율95_hi"]), (r["특이도95_lo"], r["특이도95_hi"]) = boot_ci(alarm)
            r["고령자_특이도@95"] = 1 - alarm[(y == 0) & (te.group == "SE").to_numpy()].mean()
            detail[name] = alarm
    rows.append(r)
test = pd.DataFrame(rows).set_index("후보")
print("\n== Test (임계값은 Training 교차검증에서 고정) ==")
print(test.round(3).to_string())

# 재현율 95% 임계값에서 Test 오류 내역 — 로지스틱 회귀와 LightGBM
for name in ["로지스틱 회귀", "LightGBM"]:
    a = detail[name]
    fa = te[(y == 0) & a].activity.value_counts()
    mi = te[(y == 1) & ~a].activity.value_counts()
    print(f"\n[{name}] 오경보 {int(((y == 0) & a).sum())}건:",
          {f"{k} {ACTIVITY_NAMES[k]}": v for k, v in fa.head(5).items()})
    print(f"[{name}] 놓친 낙상 {int(((y == 1) & ~a).sum())}건:",
          {f"{k} {ACTIVITY_NAMES[k]}": v for k, v in mi.head(5).items()})

# ── 교차검증 성능 + 속도와 합치기
def cv_stat(col, target):
    vals = []
    for _, g in cv.groupby("fold"):
        fpr_spec = 1 - (g[col] >= threshold_at_recall(g.is_fall, g[col], target))[g.is_fall == 0].mean()
        vals.append(fpr_spec)
    return np.mean(vals), np.std(vals, ddof=1)


sp2, sp1 = pd.read_pickle("speed_v2.pkl"), pd.read_pickle("speed_v1.pkl")
lat = pd.concat([sp2["A 컴파일 중앙값(µs)"], sp2["B ONNX 중앙값(µs)"]], axis=1).min(axis=1)
lat_how = np.where(sp2["A 컴파일 중앙값(µs)"].fillna(np.inf) <= sp2["B ONNX 중앙값(µs)"].fillna(np.inf), "컴파일", "ONNX")
summary = pd.DataFrame({
    "지연(µs)": lat.reindex(test.index),
    "측정": pd.Series(lat_how, index=sp2.index).reindex(test.index),
    "크기(KB)": sp1["모델 크기(KB)"].reindex(test.index),
    "CV 특이도@95": [cv_stat(scores[n][0], 0.95)[0] for n in test.index],
    "CV ±": [cv_stat(scores[n][0], 0.95)[1] for n in test.index],
    "Test AUC": test.Test_AUC,
    "Test 재현율@95": test["재현율@95"],
    "Test 특이도@95": test["특이도@95"],
    "특이도 95%구간": [f"{a:.3f}~{b:.3f}" for a, b in zip(test.특이도95_lo, test.특이도95_hi)],
    "재현율 95%구간": [f"{a:.3f}~{b:.3f}" for a, b in zip(test.재현율95_lo, test.재현율95_hi)],
})
print("\n== 종합 ==")
print(summary.round(3).to_string())
summary.to_csv("results_summary.csv")
test.to_csv("results_test_detail.csv")
