"""3단계 보충 — 원신호 2초(앞 1초 + 판정 구간)를 그대로 넣는 작은 1차원 합성곱 신경망(지도). 03과 같은 5묶음."""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from evaluation import summarize, threshold_at_recall
from sisfall import ACTIVITY_NAMES, EVENT_LEN, FS, PRE_LEN, list_files, load_signal

pd.set_option("display.width", 200)
CTX = PRE_LEN + EVENT_LEN  # 401샘플 = 2초
GYRO_SCALE = 500.0         # °/s 를 가속도(g)와 비슷한 범위로. 고정 상수라 데이터에서 추정하지 않는다


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(6, 16, 9, stride=2), nn.ReLU(),
            nn.Conv1d(16, 32, 7, stride=2), nn.ReLU(),
            nn.Conv1d(32, 32, 5, stride=2), nn.ReLU(),
        )
        self.head = nn.Linear(64, 1)

    def forward(self, x):  # x: (배치, 6채널, 401)
        h = self.conv(x)
        return self.head(torch.cat([h.mean(-1), h.amax(-1)], 1)).squeeze(1)


def to_input(c: np.ndarray) -> np.ndarray:
    """(401, 6) 원신호 → (6, 401) float32, 자이로만 고정 배율로 축소."""
    x = c.astype(np.float32).T.copy()
    x[3:] /= GYRO_SCALE
    return x


def train(X, y, seed=0, epochs=30):
    torch.manual_seed(seed)
    m = SmallCNN()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y.astype(np.float32))
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):  # 검증셋으로 조기 종료하지 않는다(평가 묶음을 학습에 쓰지 않기 위해 고정 횟수)
        m.train()
        for idx in torch.randperm(len(Xt), generator=g).split(64):
            opt.zero_grad()
            nn.functional.binary_cross_entropy_with_logits(m(Xt[idx]), yt[idx]).backward()
            opt.step()
    return m.eval()


def predict(m, X):
    with torch.inference_mode():
        return torch.sigmoid(m(torch.from_numpy(X))).numpy()


if __name__ == "__main__":
    meta = list_files()
    wins, pre = [], []
    for p in meta.path:
        x = load_signal(p)
        pre.append((len(x) - EVENT_LEN) / FS)
        wins.append(x[-CTX:] if len(x) >= CTX else None)
    meta["pre_s"] = pre
    keep = (meta.pre_s >= 1).to_numpy()
    np.savez_compressed("windows_v2.npz", X=np.stack([w for w, k in zip(wins, keep) if k]),
                        idx=np.flatnonzero(keep))  # 2초 원신호, 05 속도 측정에서 재사용

    # 03과 같은 사람·같은 묶음을 쓰는지 확인
    s = pd.read_pickle("cv_scores_v2.pkl")
    trm = meta[keep & (meta.split == "Training").to_numpy()].reset_index(drop=True)
    assert (trm[["subject", "activity", "rep"]].to_numpy() == s[["subject", "activity", "rep"]].to_numpy()).all()
    X = np.stack([to_input(w) for w, k, sp in zip(wins, keep, meta.split) if k and sp == "Training"])
    y, fold = s.is_fall.to_numpy(), s.fold.to_numpy()

    torch.set_num_threads(4)
    s["s_cnn"] = np.nan
    for k in range(5):
        m = train(X[fold != k], y[fold != k])
        s.loc[fold == k, "s_cnn"] = predict(m, X[fold == k])
        print(f"묶음 {k} 완료")

    methods = {"s_logreg": "지도: 로지스틱 회귀", "s_lgbm": "지도: LightGBM", "s_cnn": "지도: 1D CNN (원신호)"}
    print("\n== 비교 (5묶음 평균 ± 표준편차) ==")
    print(pd.DataFrame([summarize(s, c) for c in methods]).set_axis(methods.values()).to_string())

    adl, fall = s[s.is_fall == 0], s[s.is_fall == 1]
    print("\n== 재현율 95%에서 동작별 오경보율 (상위) ==")
    fa = pd.DataFrame({n: adl.assign(a=adl[c] >= threshold_at_recall(s.is_fall, s[c], 0.95)).groupby("activity").a.mean()
                       for c, n in methods.items()})
    fa["동작"] = fa.index.map(ACTIVITY_NAMES)
    print(fa.sort_values(methods["s_cnn"], ascending=False).head(6).round(2).to_string())
    print("\n== 재현율 95%에서 낙상 종류별 놓친 비율 (상위) ==")
    ms = pd.DataFrame({n: fall.assign(m=fall[c] < threshold_at_recall(s.is_fall, s[c], 0.95)).groupby("activity").m.mean()
                       for c, n in methods.items()})
    ms["동작"] = ms.index.map(ACTIVITY_NAMES)
    print(ms.sort_values(methods["s_cnn"], ascending=False).head(6).round(2).to_string())
    s.to_pickle("cv_scores_v3.pkl")

    final = train(X, y)  # 속도 측정용: Training 전체로 학습한 최종 모델
    torch.save(final.state_dict(), "cnn_final.pt")
