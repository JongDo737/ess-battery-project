"""초기 100사이클 ΔQ·운영 조건. 타깃과 관측 총 길이는 모델 입력에 넣지 않는다."""
import re
import numpy as np
import pandas as pd
from .preprocess import DQ_FEATURES, FULL_FEATURES, PROCESSED_CSV

# 예: 4C(80%)-4C, 4.8C(80%)-4.8C-newstructure, 3.6C(80%)-3.6C(SLOWCYCLE
STEP_RE = re.compile(r"([\d.]+)C\((\d+)%\)-([\d.]+)C", re.I)
VAR_RE = re.compile(r"VarCharge-([\d.]+)C", re.I)


def parse_policy(policy: str) -> dict:
    """충전 정책 문자열에서 1단계 C-rate, 전환 SOC, 2단계 C-rate를 읽는다."""
    text = "" if policy is None else str(policy)
    upper = text.upper()
    parsed = {
        "c1": np.nan,
        "soc": np.nan,
        "c2": np.nan,
        "is_slowcycle": int("SLOWCYCLE" in upper),
        "is_newstructure": int("NEWSTRUCTURE" in upper),
        "is_varcharge": int("VARCHARGE" in upper),
    }
    step = STEP_RE.search(text)
    if step:
        parsed["c1"] = float(step.group(1))
        parsed["soc"] = float(step.group(2))
        parsed["c2"] = float(step.group(3))
        return parsed
    var = VAR_RE.search(text)
    if var:
        # 가변 충전은 C-rate 하나만 있고 SOC·2단계는 비워 둔다.
        parsed["c1"] = float(var.group(1))
    return parsed



def load_cells(path=PROCESSED_CSV):
    cells = pd.read_csv(path)
    cells = pd.concat([cells, cells.charging_policy.map(parse_policy).apply(pd.Series)], axis=1)
    needed = ["batch", "cell_id", "cycle_life", "label_status"] + FULL_FEATURES
    if not set(needed).issubset(cells.columns):
        raise ValueError("python -m src.preprocess로 감사된 CSV를 생성하세요.")
    if not np.isfinite(cells[DQ_FEATURES].to_numpy()).all():
        raise ValueError("모든 셀의 100−10사이클 ΔQ가 필요합니다.")
    provided = cells.label_status == "provided"
    if not np.isfinite(cells.loc[provided, "cycle_life"]).all() or (cells.loc[provided, "cycle_life"] <= 0).any():
        raise ValueError("유효 수명 라벨은 유한한 양수여야 합니다.")
    if cells.loc[~provided, "cycle_life"].notna().any():
        raise ValueError("결측·검열 타깃을 수명 라벨로 채우지 마세요.")
    if cells.cell_id.duplicated().any():
        raise ValueError("중복 셀 ID")
    return cells


def feature_matrix(frame, feature_set="raw3"):
    columns = ["dq_var"] if feature_set == "logvar" else FULL_FEATURES if feature_set == "full" else DQ_FEATURES
    matrix = frame[columns].to_numpy(dtype=float)
    if feature_set in ("log3", "logvar"):
        matrix = np.log10(np.maximum(np.abs(matrix), 1e-15))
    return matrix
