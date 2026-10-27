"""SisFall_Task specified 로더 — 원본은 읽기만 하고, 정제 규칙은 읽는 시점에 적용한다.

정제 규칙
1. 주 가속도계는 ADXL345(±16g). MMA8451Q(±8g)는 낙상 파일 15%에서 상한에 걸려 쓰지 않는다
2. 100Hz로 기록된 파일(Test/SA01 D01~D06, 12개)은 선형보간으로 200Hz에 맞춘다
3. 피험자는 파일명이 아니라 폴더명으로 정한다 (Training/SA15 안의 D17_SE15_* 5개는 파일명 오타)
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

# 데이터 위치: 환경 변수 SISFALL_DIR, 없으면 이 폴더의 data/SisFall_Task specified
RAW_DIR = Path(os.environ.get("SISFALL_DIR", Path(__file__).parent / "data" / "SisFall_Task specified"))
FS = 200
EVENT_LEN = FS + 1  # FallCheck=1 구간은 항상 파일의 마지막 201샘플(1초)

ACC = ["ADXL345_Acc_X", "ADXL345_Acc_Y", "ADXL345_Acc_Z"]
GYRO = ["ITG3200_Gyro_X", "ITG3200_Gyro_Y", "ITG3200_Gyro_Z"]

# 동작 정의 출처: Sucerquia et al., "SisFall: A Fall and Movement Dataset", Sensors 2017
ACTIVITY_NAMES = {
    "D01": "천천히 걷기", "D02": "빨리 걷기", "D03": "천천히 조깅", "D04": "빨리 조깅",
    "D05": "계단 천천히 오르내리기", "D06": "계단 빨리 오르내리기",
    "D07": "중간 높이 의자 천천히 앉았다 일어서기", "D08": "중간 높이 의자 빨리 앉았다 일어서기",
    "D09": "낮은 의자 천천히 앉았다 일어서기", "D10": "낮은 의자 빨리 앉았다 일어서기",
    "D11": "일어서려다 의자에 털썩 주저앉기", "D12": "앉았다 천천히 눕고 다시 앉기",
    "D13": "앉았다 빨리 눕고 다시 앉기", "D14": "누운 채 옆으로 돌아눕기",
    "D15": "무릎 굽혀 쪼그렸다 일어서기", "D16": "무릎 펴고 허리 숙였다 일어서기",
    "D17": "차에 타고 내리기", "D18": "걷다가 비틀거리기", "D19": "가볍게 점프",
    "F01": "걷다 미끄러져 앞으로", "F02": "걷다 미끄러져 뒤로", "F03": "걷다 미끄러져 옆으로",
    "F04": "걷다 걸려 앞으로", "F05": "조깅 중 걸려 앞으로", "F06": "걷다 실신해 수직으로",
    "F07": "걷다 실신해 탁자 짚으며", "F08": "일어서다 앞으로", "F09": "일어서다 옆으로",
    "F10": "앉다가 앞으로", "F11": "앉다가 뒤로", "F12": "앉다가 옆으로",
    "F13": "앉은 채 실신해 앞으로", "F14": "앉은 채 실신해 뒤로", "F15": "앉은 채 실신해 옆으로",
}


def list_files(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    rows = []
    for f in sorted(raw_dir.glob("*/*/*.csv")):
        activity, _, rep = f.stem.split("_")
        subject = f.parent.name
        rows.append(dict(
            path=f, split=f.parent.parent.name, subject=subject,
            group=subject[:2],  # SA 청년 / SE 고령자
            activity=activity, rep=rep, is_fall=int(activity.startswith("F")),
        ))
    if not rows:
        raise FileNotFoundError(f"{raw_dir} 에서 데이터를 찾지 못했습니다. README의 '데이터 준비'를 보세요.")
    return pd.DataFrame(rows)


def load_signal(path: Path) -> np.ndarray:
    """(n, 6) 배열을 돌려준다: 가속도 3축(g) + 자이로 3축(°/s), 200Hz."""
    df = pd.read_csv(path)
    x = df[ACC + GYRO].to_numpy(np.float64)
    if df["FallCheck"].sum() < EVENT_LEN // 2 + 10:  # 1초가 101샘플 → 100Hz 파일
        t_old = np.arange(len(x)) / 100
        t_new = np.arange(0, t_old[-1] + 1e-9, 1 / FS)
        x = np.column_stack([np.interp(t_new, t_old, x[:, j]) for j in range(x.shape[1])])
    return x


def event_window(x: np.ndarray) -> np.ndarray:
    return x[-EVENT_LEN:]


def basic_features(w: np.ndarray) -> dict:
    """판정 구간 하나에서 크기 기반 지표를 뽑는다."""
    acc = np.linalg.norm(w[:, :3], axis=1)
    gyro = np.linalg.norm(w[:, 3:], axis=1)
    k = acc.argmax()
    return dict(
        acc_max=acc[k],              # 충격 크기
        acc_min=acc.min(),
        acc_std=acc.std(),
        gyro_max=gyro.max(),         # 회전 속도
        ff_min=acc[: k + 1].min(),   # 최고점 이전 최저 가속도 = 자유낙하 깊이
    )


PRE_LEN = FS  # 판정 구간 앞 1초. 낙상 파일은 최소 1.5초가 있어 모두 채워진다


def shape_features(x: np.ndarray) -> dict:
    """판정 구간 + 앞 1초(총 2초)에서 모양 지표를 뽑는다. 판정 순간 이후 데이터는 쓰지 않는다."""
    c = x[-(PRE_LEN + EVENT_LEN):]
    acc = np.linalg.norm(c[:, :3], axis=1)
    pre, ev = acc[:PRE_LEN], acc[PRE_LEN:]
    k = PRE_LEN + ev.argmax()  # 최고점 위치

    g0 = c[: FS // 4, :3].mean(0)   # 맥락 시작 0.25초의 평균 가속도 → 충격 전 중력 방향
    g1 = c[-FS // 10:, :3].mean(0)  # 마지막 0.1초의 평균 가속도
    cos = g0 @ g1 / (np.linalg.norm(g0) * np.linalg.norm(g1))

    w = c[:, 3:]                                       # 각속도(°/s)
    v = np.cumsum((acc[k - FS:k] - 1.0) * 9.81) / FS   # 최고점 직전 1초의 수직 속도 근사(m/s)
    return dict(
        peak_ratio=ev.max() / pre.max(),   # 직전 1초의 최대 충격 대비 몇 배인가 → 조깅·계단 같은 반복 충격 거르기
        pre_acc_std=pre.std(),             # 직전 1초의 활동량
        tilt_change=np.degrees(np.arccos(np.clip(cos, -1, 1))),  # 자세(중력 방향)가 몇 도 바뀌었나
        rot_net=np.linalg.norm(w.sum(0) / FS),         # 2초 동안의 순 회전각 → 한 방향으로 넘어갔나
        rot_abs=np.linalg.norm(w, axis=1).sum() / FS,  # 2초 동안 돈 총 각도 (왕복 포함)
        vel_down=-v.min(),                 # 충격 직전 최대 하강 속도
    )


def load_table(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    meta = list_files(raw_dir)
    rows = []
    for p in meta.path:
        x = load_signal(p)
        pre_s = (len(x) - EVENT_LEN) / FS
        row = dict(pre_s=pre_s, **basic_features(event_window(x)))
        if len(x) >= PRE_LEN + EVENT_LEN:
            row.update(shape_features(x))
        rows.append(row)
    return pd.concat([meta, pd.DataFrame(rows)], axis=1)
