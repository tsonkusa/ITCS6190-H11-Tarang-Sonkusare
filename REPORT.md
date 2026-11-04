# Hands-on L11: Machine Learning with MLlib

**Name:** Tarang Sonkusare
**Student ID / Seed:** 801352372
**Email:** tsonkusa@charlotte.edu

## 1. How I Ran the Lab

I used GitHub Codespaces with Spark 4.2.0, one master, and two workers.

```bash
docker compose -f docker-compose.codespaces.yml up -d

docker cp main.py spark-master:/opt/spark/work-dir/

docker exec -it spark-master /opt/spark/bin/spark-submit \
  --master spark://spark-master:7077 \
  /opt/spark/work-dir/main.py \
  task1 /opt/spark/work-dir/shared 801352372

docker exec -it spark-master /opt/spark/bin/spark-submit \
  --master spark://spark-master:7077 \
  /opt/spark/work-dir/main.py \
  task2 /opt/spark/work-dir/shared 801352372
```

## 2. Results

### Task 1: Training and Evaluation

- Training rows: 5,711
- Test rows: 1,435

| Model | RMSE | R² |
|---|---:|---:|
| Baseline | 400.2 | 0.087 |
| Full Pipeline | 387.8 | 0.143 |

First five predictions:

| Neighbourhood | Room Type | Bedrooms | Price | Prediction |
|---|---|---:|---:|---:|
| Bayview | Private room | 1 | 70 | 24.2 |
| Bernal Heights | Entire home/apt | 2 | 220 | 272.3 |
| Bernal Heights | Private room | 1 | 128 | -96.3 |
| Bernal Heights | Private room | 1 | 100 | 147.0 |
| Bernal Heights | Entire home/apt | 1 | 250 | 148.9 |

Saved model: `shared-folder/models/full`

### Task 2: Loading and Scoring

I loaded the saved PipelineModel instead of training it again.

- Loaded model RMSE: 387.8
- Test rows: 1,435
- Scored rows: 1,434

The first five predictions were identical to Task 1: 24.2, 272.3, -96.3, 147.0, and 148.9.

The loaded pipeline contained:

1. StringIndexerModel
2. OneHotEncoderModel
3. VectorAssembler
4. LinearRegressionModel

## 3. Questions

### Q1. How did the baseline and full models compare?

The full model performed better. RMSE decreased from 400.2 to 387.8 (about 3.1%), while R² increased from 0.087 to 0.143. The additional features helped, but the model still explains only 14.3% of the variation in prices.

Since Airbnb prices range from $10 to $10,000, expensive listings and outliers can increase RMSE significantly. Different seeds also produce different train/test splits, which changes the results. My seed gave 5,711 training rows and 1,435 test rows, compared with the lecture's 5,780 and 1,366.

I would first tune `regParam` using `CrossValidator` and `ParamGridBuilder` to see whether regularization improves performance.

### Q2. What is saved in the PipelineModel?

The saved model contains a `metadata/` folder and a `stages/` folder. The stages include:

- `StringIndexerModel`: fitted category-to-index mappings.
- `OneHotEncoderModel`: fitted encoding information.
- `VectorAssembler`: combines features into a vector.
- `LinearRegressionModel`: fitted regression coefficients and intercept.

I saved the entire PipelineModel so that future predictions use the same preprocessing and feature mappings as training. Saving only the regression model would not preserve those transformations.

Model folder structure:

```text
models/full/
├── metadata/
└── stages/
    ├── 0_StringIndexerModel_...
    ├── 1_OneHotEncoderModel_...
    ├── 2_VectorAssembler_...
    └── 3_LinearRegressionModel_...
```

### Q3. Why were fewer rows scored, and how would streaming work?

Task 2 started with 1,435 test rows but produced 1,434 predictions. One row was skipped because `StringIndexer` used `handleInvalid="skip"`, which filters invalid or unseen categorical values.

Task 2 only loaded and transformed the data; it did not call `fit()` again.

For streaming predictions, I would use `readStream` instead of a batch read and `writeStream` instead of a batch write, with an appropriate sink and checkpoint location. The saved PipelineModel could still be used for scoring without retraining.

## 4. Problems and Fixes

**Missing NumPy:** My first run failed with:

```text
ModuleNotFoundError: No module named 'numpy'
```

My initial installation command also failed because `/usr/local/bin/python` did not exist. I checked the Python path and installed NumPy and SciPy in all three Spark containers:

```bash
for container in spark-master spark-worker-1 spark-worker-2; do
  docker exec -u root "$container" \
    python3 -m pip install numpy scipy
done
```

This installed NumPy 2.2.6 and SciPy 1.15.3.

**Solver warning:** During Task 1, Spark reported:

```text
Cholesky solver failed due to singular covariance matrix.
Retrying with Quasi-Newton solver.
```

Spark automatically switched solvers and completed training successfully, so no code change was needed.

Both tasks ran successfully afterward, and Task 2 reused the saved model with the same RMSE of 387.8.

## 5. Conclusion

The full MLlib pipeline performed better than the baseline. I successfully trained and saved the complete pipeline, loaded it for Task 2, and generated 1,434 predictions without retraining. Both tasks completed using my assigned seed.
