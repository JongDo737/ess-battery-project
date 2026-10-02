"""Batch 1 학습·정책 Hold-out 선택, 제공된 라벨의 Batch 2·3 평가."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .preprocess import OUTPUT_CSV, PAPER_MAPE, SEED, grouped_cv, policy_holdout
from .features import feature_matrix, load_cells


def clip_life(prediction):
    return np.maximum(np.asarray(prediction, dtype=float), 1.0)


def mape(actual, prediction):
    actual = np.asarray(actual, dtype=float)
    if not np.isfinite(actual).all() or (actual <= 0).any():
        raise ValueError("MAPE에는 제공된 양수 수명 라벨만 사용할 수 있습니다.")
    return float(np.mean(np.abs(actual - clip_life(prediction)) / actual) * 100)


def score_mape(estimator, X, y):
    return -mape(y, estimator.predict(X)) / 100


def inverse_log10(values):
    return np.power(10.0, values)


def linear_net(log_target=False):
    pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), ElasticNet(max_iter=20000))
    regressor = TransformedTargetRegressor(regressor=pipeline, func=np.log10, inverse_func=inverse_log10) if log_target else pipeline
    prefix = "regressor__" if log_target else ""
    return GridSearchCV(
        regressor,
        {prefix + "elasticnet__alpha": np.logspace(-4, 2, 15),
         prefix + "elasticnet__l1_ratio": [0.1, 0.5, 0.9]},
        scoring=score_mape, n_jobs=1, error_score="raise",
    )


def candidate_models():
    models = {"DummyMean": (make_pipeline(SimpleImputer(), DummyRegressor()), "raw3")}
    # 후보 변환은 고정한다. Test 결과로 파라미터를 탐색하지 않는다.
    for feature_set in ["raw3", "logvar", "log3"]:
        for log_target in [False, True]:
            name = f"ElasticNet_{feature_set}" + ("_logtarget" if log_target else "")
            models[name] = (linear_net(log_target), feature_set)
    models["RandomForest"] = (make_pipeline(SimpleImputer(), RandomForestRegressor(
        n_estimators=300, max_depth=4, min_samples_leaf=3, random_state=SEED, n_jobs=1)), "full")
    from xgboost import XGBRegressor
    models["XGBoost"] = (make_pipeline(SimpleImputer(), XGBRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=3, objective="reg:squarederror",
        random_state=SEED, n_jobs=1)), "full")
    return models


def fit_model(model, X, y, groups):
    if isinstance(model, GridSearchCV):
        model.set_params(cv=grouped_cv(groups))
        return model.fit(X, y, groups=groups)
    return model.fit(X, y)


def cv_mape(model, X, y, groups):
    scores = []
    for train_idx, valid_idx in grouped_cv(groups).split(X, y, groups):
        if set(groups[train_idx]) & set(groups[valid_idx]):
            raise ValueError("외부 CV 정책 누수")
        fitted = fit_model(clone(model), X[train_idx], y[train_idx], groups[train_idx])
        scores.append(mape(y[valid_idx], fitted.predict(X[valid_idx])))
    return float(np.mean(scores))


def official_rows(train_cv: float, valid: float, test: float) -> pd.DataFrame:
    """학습·검증·테스트 MAPE. Gap = 뒤 − 앞, 양수면 오차가 커진 쪽이다."""
    return pd.DataFrame(
        [
            {
                "구분": "Train (Batch 1 CV)",
                "MAPE (%)": round(train_cv, 2),
                "비고": "학습 분할 안 정책 GroupKFold 평균",
            },
            {
                "구분": "Valid (Batch 1 Hold-out)",
                "MAPE (%)": round(valid, 2),
                "비고": "동일 프로토콜이 train/valid에 갈라지지 않음",
            },
            {
                "구분": "Test (Batch 2)",
                "MAPE (%)": round(test, 2),
                "비고": "다른 배치",
            },
            {
                "구분": "Gap (Train-Valid)",
                "MAPE (%)": round(valid - train_cv, 2),
                "비고": "(+) : 과적합 의심",
            },
            {
                "구분": "Gap (Valid-Test)",
                "MAPE (%)": round(test - valid, 2),
                "비고": "(+) : 배치간 일반화 저하 의심",
            },
            {
                "구분": "Gap (Target-Test)",
                "MAPE (%)": round(test - PAPER_MAPE, 2),
                "비고": "Target : 원논문 9.1%",
            },
        ]
    )


def official_b3_rows(train_cv: float, valid: float, test: float, extra: float) -> pd.DataFrame:
    """Batch 2와 Batch 3을 한 표에서 비교한다."""
    return pd.DataFrame(
        [
            {
                "구분": "Train (Batch 1 CV)",
                "MAPE (%)": round(train_cv, 2),
                "비고": "학습 분할 안 정책 GroupKFold 평균",
            },
            {
                "구분": "Valid (Batch 1 Hold-out)",
                "MAPE (%)": round(valid, 2),
                "비고": "동일 프로토콜이 train/valid에 갈라지지 않음",
            },
            {
                "구분": "Test (Batch 2)",
                "MAPE (%)": round(test, 2),
                "비고": "다른 배치",
            },
            {
                "구분": "Gap (Train-Valid)",
                "MAPE (%)": round(valid - train_cv, 2),
                "비고": "(+) : 과적합 의심",
            },
            {
                "구분": "Gap (Valid-Test)",
                "MAPE (%)": round(test - valid, 2),
                "비고": "(+) : 배치간 일반화 저하 의심",
            },
            {
                "구분": "Gap (Target-Test)",
                "MAPE (%)": round(test - PAPER_MAPE, 2),
                "비고": "Target : 원논문 9.1%",
            },
            {
                "구분": "Test (Batch 3)",
                "MAPE (%)": round(extra, 2),
                "비고": "수명이 더 긴 배치",
            },
            {
                "구분": "Gap (Batch2-Batch3)",
                "MAPE (%)": round(extra - test, 2),
                "비고": "두 테스트 배치 비교",
            },
            {
                "구분": "Gap (Target-Test)",
                "MAPE (%)": round(extra - PAPER_MAPE, 2),
                "비고": "Batch 3 기준, 논문 보고와의 차이",
            },
        ]
    )



def main(verbose=True):
    cells = load_cells()
    eligible = cells[cells.label_status == "provided"].copy()
    if eligible.cycle_life.isna().any():
        raise ValueError("provided 라벨에 결측이 있습니다.")
    train, valid = policy_holdout(eligible[eligible.batch == "Batch 1"])
    test = eligible[eligible.batch == "Batch 2"]
    extra = eligible[eligible.batch == "Batch 3"]
    groups = train.charging_policy.to_numpy()
    y_train = train.cycle_life.to_numpy()
    fitted = {}; rows = []
    # 모델·파라미터 선택에는 Batch 1만 사용한다. Test 예측은 선택 완료 후 수행한다.
    for name, (model, feature_set) in candidate_models().items():
        X_train = feature_matrix(train, feature_set)
        train_cv = cv_mape(model, X_train, y_train, groups)
        final = fit_model(clone(model), X_train, y_train, groups)
        valid_pred = clip_life(final.predict(feature_matrix(valid, feature_set)))
        valid_error = mape(valid.cycle_life, valid_pred)
        fitted[name] = (final, feature_set, train_cv, valid_error, valid_pred)
        rows.append({"model": name, "train_cv_mape": train_cv, "valid_mape": valid_error})
    selected = min((name for name in fitted if name != "DummyMean"), key=lambda name: fitted[name][3])
    model, feature_set, train_cv, valid_error, valid_pred = fitted[selected]
    test_pred = clip_life(model.predict(feature_matrix(test, feature_set)))
    extra_pred = clip_life(model.predict(feature_matrix(extra, feature_set)))
    test_error, extra_error = mape(test.cycle_life, test_pred), mape(extra.cycle_life, extra_pred)
    # 선택 이후 후보별 Test 숫자는 진단용으로만 계산한다.
    comparison = pd.DataFrame(rows)
    comparison["test_b2_mape"] = [mape(test.cycle_life, fitted[name][0].predict(feature_matrix(test, fitted[name][1]))) for name in comparison.model]
    comparison["test_b3_mape"] = [mape(extra.cycle_life, fitted[name][0].predict(feature_matrix(extra, fitted[name][1]))) for name in comparison.model]
    report = official_b3_rows(train_cv, valid_error, test_error, extra_error)
    report.loc[report["구분"] == "Test (Batch 2)", "비고"] = f"유효 라벨 {len(test)}셀 / 전체 {len(cells[cells.batch == 'Batch 2'])}셀"
    report.loc[report["구분"] == "Test (Batch 3)", "비고"] = f"유효 라벨 {len(extra)}셀 / 전체 {len(cells[cells.batch == 'Batch 3'])}셀"
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(OUTPUT_CSV, index=False)
    predictions = []
    for frame, prediction, split in [(valid, valid_pred, "Valid (Batch 1 Hold-out)"), (test, test_pred, "Test (Batch 2)"), (extra, extra_pred, "Test (Batch 3)")]:
        part = frame[["cell_id", "batch", "charging_policy", "cycle_life"]].copy()
        part["prediction"] = prediction
        part["abs_pct_error"] = np.abs(part.cycle_life - prediction) / part.cycle_life * 100
        part["split"] = split
        predictions.append(part)
    result = {"selected_model": selected, "model": model, "feature_set": feature_set,
              "performance": report, "comparison": comparison,
              "predictions": pd.concat(predictions, ignore_index=True),
              "cells": cells, "train": train, "valid": valid}
    if verbose:
        print("선택 모델:", selected)
        print("분할:", len(train), len(valid), len(test), len(extra))
        print(report.to_string(index=False))
        print(comparison.round(2).to_string(index=False))
    return result


if __name__ == "__main__":
    main()
