"""2단계 통계·규칙 방법 3종 비교 — Training 안에서 사람 단위 5묶음 교차검증."""
import numpy as np
import pandas as pd
from sklearn.covariance import EmpiricalCovariance
from sklearn.metrics import roc_auc_score

from evaluation import add_folds, summarize, threshold_at_recall
from sisfall import ACTIVITY_NAMES, load_table

pd.set_option("display.width", 200)

df = load_table()
df.drop(columns="path").to_pickle("event_features_v1.pkl")
tr = add_folds(df[df.split == "Training"].reset_index(drop=True))
print("묶음별 사람 수:", tr.groupby("fold").subject.nunique().to_dict(),
      " 낙상 수:", tr.groupby("fold").is_fall.sum().to_dict())

# ── 방법 1: 기준선 — 최고 가속도 임계값 (학습할 것이 없음)
tr["s_baseline"] = tr.acc_max

# ── 방법 2: 자유낙하 → 충격 순서 규칙
#    최고점 이전에 |a|가 T_ff 아래로 떨어진 적이 있을 때만 최고 가속도를 점수로 쓰고, 없으면 0
FF_GRID = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
print("\n== 자유낙하 기준별: 해당 구간이 있는 비율 (Training 전체) ==")
print(pd.DataFrame({t: {"낙상": (tr[tr.is_fall == 1].ff_min < t).mean(),
                        "일상동작": (tr[tr.is_fall == 0].ff_min < t).mean(),
                        "점프(D19)": (tr[tr.activity == "D19"].ff_min < t).mean()}
                    for t in FF_GRID}).round(3).to_string())


def rule_score(d, t_ff):
    return np.where(d.ff_min < t_ff, d.acc_max, 0.0)


tr["s_rule"] = np.nan
chosen = {}
for k in range(5):
    fit, va = tr.fold != k, tr.fold == k
    aucs = {t: roc_auc_score(tr.loc[fit, "is_fall"], rule_score(tr[fit], t)) for t in FF_GRID}
    chosen[k] = max(aucs, key=aucs.get)
    tr.loc[va, "s_rule"] = rule_score(tr[va], chosen[k])
print("묶음별로 고른 자유낙하 기준(g):", chosen)

# ── 방법 3: 마할라노비스 거리 — 일상동작 분포(평균·공분산)에서 얼마나 먼가
#    acc_range(=max-min)는 넣지 않는다. 다른 두 지표의 정확한 선형결합이라 공분산이 특이행렬이 된다
COLS = ["acc_max", "acc_min", "acc_std", "gyro_max"]
LOG_COLS = ["acc_max", "acc_std", "gyro_max"]  # 오른쪽 꼬리가 길어 로그로 펴본다


def mahalanobis_cv(d, use_log):
    X = d[COLS].copy()
    if use_log:
        X[LOG_COLS] = np.log(X[LOG_COLS])
    out = np.full(len(d), np.nan)
    for k in range(5):
        fit = ((d.fold != k) & (d.is_fall == 0)).to_numpy()  # 학습은 일상동작만
        va = (d.fold == k).to_numpy()
        out[va] = EmpiricalCovariance().fit(X[fit]).mahalanobis(X[va])
    return out


tr["s_maha_raw"] = mahalanobis_cv(tr, use_log=False)
tr["s_maha_log"] = mahalanobis_cv(tr, use_log=True)

print("\n== 비교 (5묶음 평균 ± 표준편차) ==")
methods = {"s_baseline": "기준선: 최고 가속도", "s_rule": "자유낙하→충격 규칙",
           "s_maha_raw": "마할라노비스 (원값)", "s_maha_log": "마할라노비스 (로그)"}
res = pd.DataFrame([summarize(tr, c) for c in methods]).set_axis(methods.values())
print(res.to_string())

# 재현율 95%에서 동작별 오경보율 — 묶음 결과를 모아 한 임계값으로
print("\n== 재현율 95%에서 동작별 오경보율 (오경보 상위 동작) ==")
adl = tr[tr.is_fall == 0]
fa = pd.DataFrame({
    name: adl.assign(a=adl[c] >= threshold_at_recall(tr.is_fall, tr[c], 0.95)).groupby("activity").a.mean()
    for c, name in methods.items()})
fa["동작"] = fa.index.map(ACTIVITY_NAMES)
print(fa.sort_values(methods["s_baseline"], ascending=False).head(10).round(2).to_string())
print("\n전체 오경보율:", {name: round(float(fa.loc[:, name].mul(adl.activity.value_counts()).sum() / len(adl)), 3)
                       for name in methods.values()})
