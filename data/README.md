# 데이터

`processed/cells_summary.csv`는 원본 139셀과 라벨 감사 결과를 보존한다. 모델 학습·평가는 `label_status=provided`인 119셀만 사용한다.

- Batch 1: 원본 46셀, 유효 36셀, 우측 검열 10셀.
- Batch 2: 원본 47셀, 유효 39셀, 결측 수명 8셀.
- Batch 3: 원본 46셀, 유효 44셀, 결측 수명 2셀.

`source_cycle_life`는 원본 제공 라벨이다. `cycle_life`는 학습용 라벨이며 결측·검열은 NaN이다. `n_cycles`·`last_QD`는 감사용으로만 보존하고 모델 입력에 넣지 않는다. 제공 라벨이 관측 길이+1 이상이고 마지막 QD가 0.885 Ah보다 높은 셀은 `right_censored`로 표시한다. 결측 수명은 `missing`이며 관측 종료 시점으로 채우지 않는다.

원본 `.mat` 파일은 [data.matr.io](https://data.matr.io/1/)에서 받는다. 프로젝트 루트에서 `python -m src.preprocess --data-dir /원본/폴더`를 실행한다. 기본 폴더는 `data/raw/`이며 기본 출력은 `data/processed/cells_summary.csv`다. `--output`으로 출력 경로를 변경할 수 있다.

- `2017-05-12_batchdata_updated_struct_errorcorrect.mat` — Batch 1
- `2018-02-20_batchdata_updated_struct_errorcorrect.mat` — Batch 2
- `2018-04-12_batchdata_updated_struct_errorcorrect.mat` — Batch 3

`2018-04-03 varcharge`는 사용하지 않는다. ΔQ는 summary의 실제 cycle 번호 100과 10에 대응하는 Qdlin의 차이다. 두 곡선 길이가 다르거나 해당 사이클이 없으면 중단한다. 다른 관측 창으로 대체하지 않는다.
