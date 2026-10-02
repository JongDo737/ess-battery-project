# ESS 배터리 수명 예측

초기 100사이클의 방전 곡선 차이로 배터리 `cycle_life`를 예측한다. 정확한 수명 라벨이 없는 관측 종료 시점을 수명으로 대신 쓰지 않는다.

## 프로젝트 개요

- 데이터: MIT-Stanford Battery Dataset (Severson et al., Nature Energy 2019).
- 원본: Batch 1 46셀, Batch 2 47셀, Batch 3 46셀. 총 139셀을 가공 CSV에 보존한다.
- 유효 회귀 라벨: Batch 1 36셀, Batch 2 39셀, Batch 3 44셀.
- 학습·검증: Batch 1에서 정책 단위 Hold-out, 학습 28셀·검증 8셀.
- 최종 모델: ElasticNet, 로그 변환한 ΔQ 통계량 3개 → `log10(cycle_life)` 회귀.
- 평가: Batch 2 39셀 MAPE 28.01%, Batch 3 44셀 MAPE 14.09%.

**28.01%는 Batch 2 전체 47셀의 성능이 아니라 유효 라벨 39셀의 성능이다.** 결측 수명 8셀의 정확한 MAPE는 계산할 수 없다. `2018-04-03 varcharge` 파일은 사용하지 않는다.

## 파일 구조

```text
├── data/
│   ├── README.md
│   └── processed/
│       └── cells_summary.csv
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling.ipynb
├── src/
│   ├── preprocess.py
│   ├── features.py
│   └── train.py
├── results/
│   └── model_performance.csv
├── requirements.txt
└── README.md
```

가공 CSV는 원본 파일 없이 실행하기 위해 포함한다. EDA·피처·모델 비교·셀별 예측·오류 그림은 실행 결과가 저장된 세 노트북에서 확인한다. 성능 CSV 한 파일에 Batch 2·3을 함께 보고한다.

## 환경 설정과 실행

Python 3.11에서 검증했다.

```bash
git clone https://github.com/JongDo737/ess-battery-project.git
cd ess-battery-project
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.train
```

Jupyter 환경에서 `notebooks/01_EDA.ipynb` → `02_feature_engineering.ipynb` → `03_modeling.ipynb` 순서로 실행한다. JupyterLab UI가 필요하면 `pip install jupyterlab` 후 `jupyter lab`을 사용한다. 노트북 3개와 독립 폴더에서의 CLI 실행을 확인했다.

원본 재추출:

```bash
python -m src.preprocess --data-dir /원본_mat_폴더
python -m src.train
```

원본 파일명과 라벨 감사 기준은 [data/README.md](data/README.md)에 있다. 원본 추출은 초기 summary와 10·100사이클 Qdlin만 직접 읽으며, 외부 Day 1 스크립트나 캐시에 의존하지 않는다.

## EDA와 라벨 감사

기존 추출은 원본 `cycle_life`가 결측이고 0.88 Ah에 도달하지 않은 셀에 `n_cycles`를 수명으로 채웠다. 이 때문에 관측이 101사이클에서 끝난 Batch 2-37·38이 수명 101인 셀로 평가됐다. 두 셀의 마지막 QD는 각각 약 2.1693·2.1654 Ah이며 수명 종료 라벨이 없다.

- Batch 2 결측 라벨: 22, 23, 35, 36, 37, 38, 39, 40번 셀. 8셀을 점수 계산에서 제외한다.
- Batch 3 결측 라벨: 23, 32번 셀. 2셀을 점수 계산에서 제외한다.
- Batch 1 중도 종료: 0, 1, 2, 3, 4, 8, 10, 12, 13, 22번 셀. 제공 라벨이 관측 길이+1 이상이고 마지막 QD가 0.885 Ah보다 높다. 정확한 EOL을 관측하지 못한 우측 검열로 표시하고 10셀을 학습·검증에서 제외한다.

원본 라벨은 `source_cycle_life`에 보존한다. 결측·검열 셀의 학습용 `cycle_life`는 NaN이다. 제외 규칙은 원본 라벨·종료 상태로 결정하며, 큰 예측 오차나 짧은 수명을 이유로 제외하지 않는다. 나머지 타깃은 원본의 유한한 양수 라벨을 그대로 사용한다.

저자 코드에도 중도 종료 셀 제외와 연속 실험 결합이 있다. 다만 저자 Batch 2는 `2017-06-30`, 이 프로젝트의 지정 Batch 2는 `2018-02-20`이다. 파일이 다르므로 저자 코드의 연속 셀 매핑이나 수명 보정 숫자를 이 데이터에 임의로 적용하지 않는다.

유효 라벨을 대상으로 분포·충전 정책·ΔQ 관계를 다시 분석했다. 라벨 없는 101사이클 관측 셀을 단수명 사례로 해석하지 않는다. Batch 2의 유효 수명 범위도 Batch 1과 달라 배치 이동이 남는다.

## 피처 엔지니어링과 모델 선택

- 핵심 변수: `ΔQ(V) = Qdlin(cycle=100) − Qdlin(cycle=10)`.
- 선택 피처: `log10(max(abs(dq_var), 1e−15))`, `log10(max(abs(dq_min), 1e−15))`, `log10(max(abs(dq_mean), 1e−15))`.
- 타깃: `log10(cycle_life)`. 예측 시 `10**prediction`으로 원수명 단위에 복원한다.
- 후보: Dummy, ElasticNet(원 ΔQ·로그 분산·로그 ΔQ 3개 × 원수명·로그 수명), RandomForest, XGBoost.
- 내부 튜닝: alpha 15개(10⁻⁴–10² 로그 간격), l1_ratio 0.1·0.5·0.9, 정책 GroupKFold의 MAPE.
- 외부 CV: 학습 28셀 안에서만 정책 GroupKFold. 각 외부 학습 fold 안에서 내부 튜닝을 새로 수행한다.
- 전처리: 결측 중앙값과 StandardScaler를 각 학습 fold 안에서 fit한다.
- 선택 기준: Batch 1 Hold-out MAPE. 최종 `ElasticNet_log3_logtarget`의 Valid MAPE가 5.67%로 가장 낮았다.

선택된 alpha는 0.0007196856730011522, l1_ratio는 0.1이다. 다른 후보의 테스트 점수가 더 낮더라도 테스트 결과로 재선택하지 않는다. 관측 총 길이·최종 용량·라벨 상태는 모델 입력에 넣지 않는다.

Batch 2 결과를 이미 확인한 상태에서 라벨 오류를 진단하고 전략을 수정했다. 따라서 이 결과는 **테스트를 확인한 뒤 수정한 개발 실험**이다. 새로운 블라인드 테스트나 독립 외부 검증으로 주장하지 않는다. 최종 코드의 파라미터·후보 선택에는 Batch 1만 들어간다.

## 성능 결과

MAPE는 낮을수록 좋다. Gap은 반올림 전 성능에서 뒤 구간 − 앞 구간을 계산한 뒤 소수 둘째 자리로 반올림한다. 표시된 값끼리 빼면 0.01%p 차이가 날 수 있다. Gap의 단위는 %p다.

| 구분 | MAPE (%) | 비고 |
| --- | --- | --- |
| Train (Batch 1 CV) | 9.63 | 학습 분할 안 정책 GroupKFold 평균 |
| Valid (Batch 1 Hold-out) | 5.67 | 동일 프로토콜이 train/valid에 갈라지지 않음 |
| Test (Batch 2) | 28.01 | 유효 라벨 39셀 / 전체 47셀 |
| Gap (Train-Valid) | -3.95 | (+) : 과적합 의심 |
| Gap (Valid-Test) | 22.34 | (+) : 배치간 일반화 저하 의심 |
| Gap (Target-Test) | 18.91 | Target : 원논문 9.1% |
| Test (Batch 3) | 14.09 | 유효 라벨 44셀 / 전체 46셀 |
| Gap (Batch2-Batch3) | -13.92 | 두 테스트 배치 비교 |
| Gap (Target-Test) | 4.99 | Batch 3 기준, 논문 보고와의 차이 |

Valid-Test Gap 22.34%p가 남아 배치 간 일반화 저하가 있다. Batch 3는 Batch 2보다 MAPE가 낮지만, 수명 분포와 MAPE 분모가 다르므로 이 차이만으로 피처의 배치 과적합 여부를 확정하지 않는다. 논문 9.1%는 저자 전처리 124셀·저자 분할로, 이번 유효 라벨 119셀·배치 분할과 조건이 다르다.

## 오류 분석

- Batch 2-15: `3.6C(9%)-5C`, 실제 396, 예측 635.93, APE 60.59%.
- Batch 2-18: `5.2C(50%)-4.25C`, 실제 449, 예측 674.35, APE 50.19%.
- Batch 2-6: `3.6C(9%)-5C`, 실제 393, 예측 587.43, APE 49.47%.

큰 오차가 있는 셀은 400사이클 전후 수명에서 과대 예측되는 경우다. 특히 `3.6C(9%)-5C`, `5.2C(50%)-4.25C` 정책의 셀이 반복해서 나타난다. 라벨 오류를 고쳐도 이 배치 이동은 없어지지 않았다.

기존 XGBoost 72.68%는 잘못 채운 타깃을 포함한 47셀의 값이었다. 동일한 기존 XGBoost를 원본 제공 라벨 39셀에서만 평가하면 33.59%였다. 이후 Batch 1의 검열 타깃도 제외하고 로그 피처·로그 수명 후보를 Valid 기준으로 비교해 최종 28.01%를 얻었다. 평가 대상과 모델이 모두 바뀌었으므로 72.68 → 28.01을 같은 테스트 조건에서의 순수 모델 성능 향상으로 비교하지 않는다.

## ESS 도메인 해석

초기 관측으로 셀의 상대적인 수명 위험을 선별하거나 추가 검사가 필요한 로트를 찾는 데 활용할 수 있다. Batch 2 MAPE 28.01%와 일부 셀의 큰 과대 예측을 고려하면 이 모델만으로 교체 시점을 확정할 수 없다.

실험실 LFP/흑연 고속충전 데이터이며 필드 부하·캘린더 에이징이 포함되지 않는다. 100사이클을 기다려야 하고, 결측·검열 셀의 실제 수명은 확인하지 못한다. 실 BESS 적용에는 현장 조건과 새로운 배치의 검증, 검열 데이터를 다루는 별도 설계, 예측 불확실성 평가가 필요하다.

## 참고문헌

- Severson et al. (2019). Data-driven prediction of battery cycle life before capacity degradation. Nature Energy 4, 383–391. https://doi.org/10.1038/s41560-019-0356-8
- [저자 데이터 처리 코드](https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation/blob/master/LoadData.m)
- [원본 데이터](https://data.matr.io/1/)

## 작성

신종민
