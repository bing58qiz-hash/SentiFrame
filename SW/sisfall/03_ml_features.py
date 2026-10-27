"""3단계 머신러닝 — 같은 피처로 비지도 2종 vs 지도 2종. Training 사람 단위 5묶음 교차검증."""
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluation import add_folds, summarize, threshold_at_recall
from sisfall import ACTIVITY_NAMES, load_table

pd.set_option("display.width", 220)

df = load_table()
df.drop(columns="path").to_pickle("event_features_v2.pkl")
tr = df[df.split == "Training"]
print(f"판정 구간 앞 1초 미만으로 제외: 낙상 {((tr.pre_s < 1) & (tr.is_fall == 1)).sum()}, "
      f"일상동작 {((tr.pre_s < 1) & (tr.is_fall == 0)).sum()}")
tr = add_folds(tr[tr.pre_s >= 1].reset_index(drop=True))

SIZE = ["acc_max", "acc_min", "acc_std", "gyro_max", "ff_min"]
SHAPE = ["peak_ratio", "pre_acc_std", "tilt_change", "rot_net", "rot_abs", "vel_down"]
FEATS = SIZE + SHAPE

print("\n== 모양 피처 하나씩의 AUC (Training, 0.5 아래면 작을수록 낙상) ==")
print(pd.Series({c: roc_auc_score(tr.is_fall, tr[c]) for c in SHAPE}).round(3).to_string())


def cv(fit_predict, feats=FEATS):
    out = np.full(len(tr), np.nan)
    for k in range(5):
        fit, va = tr[tr.fold != k], tr[tr.fold == k]
        out[(tr.fold == k).to_numpy()] = fit_predict(fit, va, feats)
    return out


# ── 비지도: 4묶음의 일상동작만으로 학습, 낙상 라벨은 쓰지 않는다
def iforest(fit, va, f):
    m = IsolationForest(n_estimators=300, random_state=0).fit(fit.loc[fit.is_fall == 0, f])
    return -m.score_samples(va[f])  # 클수록 이상


def autoencoder(fit, va, f):
    sc = StandardScaler().fit(fit.loc[fit.is_fall == 0, f])
    X = sc.transform(fit.loc[fit.is_fall == 0, f])
    ae = MLPRegressor(hidden_layer_sizes=(8, 3, 8), max_iter=3000, random_state=0).fit(X, X)
    Xv = sc.transform(va[f])
    return ((ae.predict(Xv) - Xv) ** 2).mean(1)  # 복원 오차


# ── 지도: 4묶음의 낙상·일상동작 라벨로 학습
def logreg(fit, va, f):
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(fit[f], fit.is_fall)
    return m.predict_proba(va[f])[:, 1]


def lgbm(fit, va, f):
    m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15,
                           random_state=0, verbose=-1).fit(fit[f], fit.is_fall)
    return m.predict_proba(va[f])[:, 1]


tr["s_baseline"] = tr.acc_max
tr["s_iforest"] = cv(iforest)
tr["s_ae"] = cv(autoencoder)
tr["s_logreg"] = cv(logreg)
tr["s_lgbm"] = cv(lgbm)
tr["s_lgbm_size"] = cv(lgbm, SIZE)  # 모양 피처의 기여를 보려고: 크기 피처만 넣은 LightGBM

methods = {"s_baseline": "기준선: 최고 가속도",
           "s_iforest": "비지도: Isolation Forest", "s_ae": "비지도: 오토인코더",
           "s_logreg": "지도: 로지스틱 회귀", "s_lgbm": "지도: LightGBM",
           "s_lgbm_size": "(참고) LightGBM 크기 피처만"}
print(f"\n== 비교 (5묶음 평균 ± 표준편차) — 낙상 {tr.is_fall.sum()}, 일상동작 {(tr.is_fall == 0).sum()} ==")
print(pd.DataFrame([summarize(tr, c) for c in methods]).set_axis(methods.values()).to_string())

print("\n== 재현율 95%에서 동작별 오경보율 (기준선 오경보 상위) ==")
adl = tr[tr.is_fall == 0]
show = ["s_baseline", "s_iforest", "s_ae", "s_logreg", "s_lgbm"]
fa = pd.DataFrame({methods[c]: adl.assign(a=adl[c] >= threshold_at_recall(tr.is_fall, tr[c], 0.95))
                   .groupby("activity").a.mean() for c in show})
fa["동작"] = fa.index.map(ACTIVITY_NAMES)
print(fa.sort_values(methods["s_baseline"], ascending=False).head(10).round(2).to_string())

print("\n== 재현율 95%에서 낙상 종류별 놓친 비율 (LightGBM 상위 5) ==")
fall = tr[tr.is_fall == 1]
t = threshold_at_recall(tr.is_fall, tr.s_lgbm, 0.95)
miss = fall.assign(m=fall.s_lgbm < t).groupby("activity").m.mean().sort_values(ascending=False).head(5)
print(pd.DataFrame({"놓친 비율": miss.round(2), "동작": miss.index.map(ACTIVITY_NAMES)}).to_string())

tr.drop(columns="path").to_pickle("cv_scores_v2.pkl")
