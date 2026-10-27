"""기준선 — 판정 구간(마지막 1초)의 단일 지표 하나로 임계값 판정. Training만 사용."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from sisfall import ACTIVITY_NAMES, EVENT_LEN, event_window, list_files, load_signal

pd.set_option("display.width", 200)

meta = list_files()
feats = []
for p in meta.path:
    w = event_window(load_signal(p))
    acc = np.linalg.norm(w[:, :3], axis=1)
    gyro = np.linalg.norm(w[:, 3:], axis=1)
    feats.append(dict(
        n_event=len(w),
        acc_max=acc.max(),                  # 충격 크기
        acc_min_neg=-acc.min(),             # 자유낙하 깊이 (부호를 뒤집어 클수록 낙상 쪽)
        acc_range=acc.max() - acc.min(),
        acc_std=acc.std(),
        gyro_max=gyro.max(),                # 회전 속도
    ))
df = pd.concat([meta, pd.DataFrame(feats)], axis=1)
assert (df.n_event == EVENT_LEN).all()
df.drop(columns="path").to_pickle("event_features_v0.pkl")

tr = df[df.split == "Training"]
cands = ["acc_max", "acc_min_neg", "acc_range", "acc_std", "gyro_max"]
print("== 단일 지표별 AUC (Training) ==")
auc = pd.Series({c: roc_auc_score(tr.is_fall, tr[c]) for c in cands}).sort_values(ascending=False)
print(auc.round(4).to_string())
best = auc.index[0]


def at_recall(y, s, targets):
    fpr, tpr, thr = roc_curve(y, s)
    out = []
    for r in targets:
        i = np.argmax(tpr >= r)
        out.append(dict(target_recall=r, threshold=thr[i], recall=tpr[i], specificity=1 - fpr[i],
                        false_alarms=int(round(fpr[i] * (y == 0).sum()))))
    return pd.DataFrame(out)


print(f"\n== 기준선 '{best}' — 재현율 수준별 특이도 (Training: 낙상 {tr.is_fall.sum()}, 일상동작 {(tr.is_fall == 0).sum()}) ==")
tab = at_recall(tr.is_fall.to_numpy(), tr[best].to_numpy(), [0.80, 0.85, 0.90, 0.95, 0.97, 0.99, 1.0])
print(tab.round(3).to_string(index=False))

t95 = tab.loc[tab.target_recall == 0.95, "threshold"].item()
tr = tr.assign(alarm=tr[best] >= t95)
adl, fall = tr[tr.is_fall == 0], tr[tr.is_fall == 1]
print(f"\n== 재현율 95% 임계값({best} >= {t95:.2f})에서 ==")
print(f"청년 일상동작 오경보율 {adl[adl.group == 'SA'].alarm.mean():.3f} / 고령자 {adl[adl.group == 'SE'].alarm.mean():.3f}")
fa = adl.groupby("activity").alarm.agg(["mean", "sum", "size"]).sort_values("mean", ascending=False)
fa["name"] = fa.index.map(ACTIVITY_NAMES)
print("\n동작별 오경보율 (상위 8):"); print(fa.head(8).round(3).to_string())
miss = fall.groupby("activity").alarm.agg(lambda a: 1 - a.mean()).rename("miss_rate").sort_values(ascending=False)
print("\n낙상 종류별 놓친 비율 (상위 5):")
print(pd.DataFrame({"miss_rate": miss, "name": miss.index.map(ACTIVITY_NAMES)}).head(5).round(3).to_string())
