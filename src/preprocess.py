"""원본 라벨 감사·피처 추출·충전 정책 분할. 관측 종료를 수명으로 대체하지 않는다."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_CSV = ROOT / "data" / "processed" / "cells_summary.csv"
OUTPUT_CSV = ROOT / "results" / "model_performance.csv"
SEED = 42
PAPER_MAPE = 9.1
DQ_FEATURES = ["dq_var", "dq_min", "dq_mean"]
FULL_FEATURES = DQ_FEATURES + ["mean_chargetime", "c1", "soc", "c2", "mean_Tavg"]
BATCHES = [
    ("Batch 1", "2017-05-12_batchdata_updated_struct_errorcorrect.mat"),
    ("Batch 2", "2018-02-20_batchdata_updated_struct_errorcorrect.mat"),
    ("Batch 3", "2018-04-12_batchdata_updated_struct_errorcorrect.mat"),
]


def _array(handle, dataset):
    import h5py
    values = dataset[()]
    if h5py.check_dtype(ref=values.dtype) is not None:
        return np.concatenate([handle[ref][()].reshape(-1) for ref in values.reshape(-1)])
    return values.reshape(-1)


def extract_cells(data_dir):
    """HDF5 원본의 summary·10/100사이클만 읽는다. 전체 cycles를 메모리에 올리지 않는다."""
    import h5py
    rows = []
    for batch_name, filename in BATCHES:
        with h5py.File(Path(data_dir) / filename, "r") as handle:
            batch = handle["batch"]
            for i in range(batch["cycle_life"].shape[0]):
                summary = handle[batch["summary"][i, 0]]
                cycles = handle[batch["cycles"][i, 0]]
                life = float(handle[batch["cycle_life"][i, 0]][()].reshape(-1)[0])
                if not np.isfinite(life) or life <= 0:
                    life = np.nan
                policy_data = handle[batch["policy_readable"][i, 0]][()].reshape(-1)
                policy = "".join(chr(int(v)) for v in policy_data)
                qd = _array(handle, summary["QDischarge"])
                cycle_numbers = _array(handle, summary["cycle"])
                vectors = []
                for number in (10, 100):
                    positions = np.flatnonzero(cycle_numbers == number)
                    if len(positions) != 1:
                        raise ValueError(f"{batch_name}-{i}: cycle {number}을 식별할 수 없습니다.")
                    index = int(positions[0])
                    vectors.append(handle[cycles["Qdlin"][index, 0]][()].reshape(-1))
                if vectors[0].shape != vectors[1].shape:
                    raise ValueError(f"{batch_name}-{i}: ΔQ 전압창 길이가 다릅니다.")
                delta = vectors[1] - vectors[0]
                if not np.isfinite(delta).all():
                    raise ValueError(f"{batch_name}-{i}: ΔQ가 유효하지 않습니다.")
                row = {
                    "batch": batch_name, "cell_id": f"{batch_name}-{i}",
                    "cycle_life": life, "source_cycle_life": life, "charging_policy": policy,
                    "n_cycles": len(qd), "last_QD": float(qd[-1]),
                    "label_status": "provided" if np.isfinite(life) else "missing",
                    "dq_var": float(np.var(delta)), "dq_min": float(np.min(delta)),
                    "dq_mean": float(np.mean(delta)),
                }
                for key, column in [("QDischarge", "mean_QD"), ("IR", "mean_IR"),
                                    ("Tavg", "mean_Tavg"), ("Tmax", "mean_Tmax"),
                                    ("chargetime", "mean_chargetime")]:
                    row[column] = float(np.nanmean(_array(handle, summary[key])[:100]))
                # 제공 라벨이 관측 길이+1이고 EOL 용량에 도달하지 않았다면 우측 검열이다.
                if np.isfinite(life) and life >= len(qd) + 1 and qd[-1] > 0.885:
                    row["cycle_life"] = np.nan
                    row["label_status"] = "right_censored"
                rows.append(row)
    return pd.DataFrame(rows)


def policy_holdout(batch1):
    split = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
    train_idx, valid_idx = next(split.split(batch1, groups=batch1.charging_policy))
    train, valid = batch1.iloc[train_idx].copy(), batch1.iloc[valid_idx].copy()
    if set(train.charging_policy) & set(valid.charging_policy):
        raise ValueError("정책 Hold-out 누수")
    return train, valid


def grouped_cv(groups):
    count = len(np.unique(groups))
    if count < 2:
        raise ValueError("정책 CV에 최소 2개 그룹이 필요합니다.")
    return GroupKFold(n_splits=min(5, count))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--output", type=Path, default=PROCESSED_CSV)
    args = parser.parse_args()
    cells = extract_cells(args.data_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cells.to_csv(args.output, index=False)
    print(cells.groupby("batch").cycle_life.agg(["size", "count"]))
    print("저장:", args.output)


if __name__ == "__main__":
    main()
