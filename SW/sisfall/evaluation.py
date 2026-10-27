"""평가 설계 — 모든 방법을 같은 조건에서 비교한다.

- Training 30명을 사람 단위로 5묶음으로 나눈다. 같은 사람은 한 묶음에만 들어간다
- 학습·파라미터 선택이 있는 방법은 나머지 4묶음으로 정하고, 남긴 1묶음에서 점수를 낸다
- 지표: 묶음별 AUC, 재현율 90/95/99%에서의 특이도 → 평균과 표준편차
- Test는 최종 후보를 고른 뒤 한 번만 쓴다
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold

RECALLS = (0.90, 0.95, 0.99)


def add_folds(df: pd.DataFrame, n_splits: int = 5, seed: int = 0) -> pd.DataFrame:
    df = df.copy()
    df["fold"] = -1
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for k, (_, va) in enumerate(cv.split(df, df.is_fall, df.subject)):
        df.iloc[va, df.columns.get_loc("fold")] = k
    return df


def threshold_at_recall(y, s, recall: float) -> float:
    _, tpr, thr = roc_curve(y, s)
    return thr[np.argmax(tpr >= recall)]


def spec_at_recall(y, s, recall: float) -> float:
    fpr, tpr, _ = roc_curve(y, s)
    return 1 - fpr[np.argmax(tpr >= recall)]


def summarize(df: pd.DataFrame, score_col: str) -> pd.Series:
    rows = []
    for _, g in df.groupby("fold"):
        y, s = g.is_fall.to_numpy(), g[score_col].to_numpy()
        rows.append({"AUC": roc_auc_score(y, s),
                     **{f"spec@{round(r * 100)}": spec_at_recall(y, s, r) for r in RECALLS}})
    m = pd.DataFrame(rows)
    return pd.Series({c: f"{m[c].mean():.3f} ± {m[c].std():.3f}" for c in m.columns}, name=score_col)
