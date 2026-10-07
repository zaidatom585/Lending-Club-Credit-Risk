# Databricks notebook source
# MAGIC %md
# MAGIC # 05 Train: default models vs. Lending Club's grades
# MAGIC
# MAGIC Trains two models on `gold_features` and tracks both in **MLflow**:
# MAGIC 1. **Logistic regression**: a transparent baseline, the traditional credit-scoring model.
# MAGIC 2. **LightGBM**: gradient-boosted trees, the standard for tabular credit data.
# MAGIC
# MAGIC Both are trained on **2007–2013**, tuned on **2014**, and judged on **2015**, then compared with
# MAGIC Lending Club's own **sub-grade** and **interest rate**, used as risk scores.
# MAGIC
# MAGIC **Metrics** (higher is better unless noted):
# MAGIC - **ROC AUC**: how well the model ranks defaulters above non-defaulters (0.5 = random).
# MAGIC - **Gini** = 2 × AUC − 1, the version banks usually report.
# MAGIC - **KS**: the largest gap between the score distributions of defaulters and non-defaulters.
# MAGIC - **PR AUC**: precision-recall area, informative when defaults are the minority.
# MAGIC - **Brier score** (lower is better): accuracy of the predicted probabilities themselves.

# COMMAND ----------

# MAGIC %pip install --quiet lightgbm

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import mlflow
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CATALOG = "workspace"
SCHEMA = "lending_club"
T = lambda name: f"{CATALOG}.{SCHEMA}.{name}"

# Must match 04_build_model_features.py
NUMERIC_FEATURES = [
    "loan_amnt", "annual_inc_log", "dti", "fico_avg", "inq_last_6mths", "delinq_2yrs",
    "open_acc", "total_acc", "pub_rec", "revol_bal_log", "revol_util", "mort_acc",
    "pub_rec_bankruptcies", "credit_history_years", "emp_length_years", "loan_to_income",
    "months_since_last_delinq", "tot_cur_bal_log", "total_rev_hi_lim_log",
    "acc_open_past_24mths", "bc_util", "num_actv_rev_tl", "percent_bc_gt_75",
]
FLAG_FEATURES = ["emp_length_missing", "never_delinquent"]
CATEGORICAL_FEATURES = ["home_ownership", "verification_status", "purpose", "addr_state", "application_type"]
FEATURES = NUMERIC_FEATURES + FLAG_FEATURES + CATEGORICAL_FEATURES

FORBIDDEN = {"grade", "sub_grade", "int_rate", "installment", "payment_to_income", "loan_status",
             "total_pymnt", "recoveries", "last_pymnt_amnt", "funded_amnt", "is_default"}
assert not (set(FEATURES) & FORBIDDEN), f"Leakage: {set(FEATURES) & FORBIDDEN}"

user = spark.sql("SELECT current_user()").first()[0]
mlflow.set_experiment(f"/Users/{user}/lending-club-default-risk")


def log_model(flavor, model, **kwargs):
    """Save a model to the active MLflow run.

    Newer MLflow versions use `name=` instead of `artifact_path=`; older ones only accept `artifact_path=`.
    """
    try:
        return flavor.log_model(model, name="model", **kwargs)
    except TypeError:
        return flavor.log_model(model, artifact_path="model", **kwargs)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load the data
# MAGIC About 620,000 rows × 30 features fits comfortably in memory as a pandas DataFrame.

# COMMAND ----------

df = spark.table(T("gold_features")).toPandas()
for c in NUMERIC_FEATURES + FLAG_FEATURES:
    df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
df["is_default"] = df["is_default"].astype(int)

# Category levels come from training data only, so validation/test can't influence them.
train_mask = df["split"] == "train"
for c in CATEGORICAL_FEATURES:
    levels = sorted(df.loc[train_mask, c].dropna().unique())
    df[c] = pd.Categorical(df[c], categories=levels)

splits = {s: df[df["split"] == s].reset_index(drop=True) for s in ["train", "validation", "test"]}
for s, d in splits.items():
    print(f"{s:<11} {len(d):>8,} loans   default rate {100 * d['is_default'].mean():5.2f}%")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluation helpers

# COMMAND ----------

SUB_GRADES = [f"{g}{n}" for g in "ABCDEFG" for n in range(1, 6)]
SUB_GRADE_RANK = {sg: i for i, sg in enumerate(SUB_GRADES)}


def evaluate(y, score, prob=None):
    fpr, tpr, _ = roc_curve(y, score)
    auc = roc_auc_score(y, score)
    out = {
        "roc_auc": auc,
        "gini": 2 * auc - 1,
        "ks": float(np.max(tpr - fpr)),
        "pr_auc": average_precision_score(y, score),
    }
    if prob is not None:
        out["brier"] = brier_score_loss(y, prob)
    return {k: round(float(v), 4) for k, v in out.items()}


results = []


def record(model_name, split, metrics):
    results.append({"model": model_name, "split": split, **metrics})

# COMMAND ----------

# MAGIC %md
# MAGIC ## Benchmark: Lending Club's own risk assessment

# COMMAND ----------

for s in ["validation", "test"]:
    d = splits[s]
    record("Lending Club sub-grade", s, evaluate(d["is_default"], d["sub_grade"].map(SUB_GRADE_RANK)))
    record("Lending Club interest rate", s, evaluate(d["is_default"], d["int_rate"]))
pd.DataFrame(results)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model 1: logistic regression
# MAGIC Missing numbers are filled with the training median (plus a "was missing" indicator), numbers are
# MAGIC standardized, and categories are one-hot encoded.

# COMMAND ----------

logreg = Pipeline([
    ("prep", ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]), NUMERIC_FEATURES),
        ("flags", "passthrough", FLAG_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=0.001), CATEGORICAL_FEATURES),
    ])),
    ("model", LogisticRegression(C=0.5, max_iter=2000)),
])


def to_object(d):
    """scikit-learn's encoder works on plain values, not pandas categories."""
    x = d[FEATURES].copy()
    for c in CATEGORICAL_FEATURES:
        x[c] = x[c].astype(object).where(x[c].notna(), "missing")
    return x


with mlflow.start_run(run_name="logistic_regression") as run:
    logreg.fit(to_object(splits["train"]), splits["train"]["is_default"])
    mlflow.log_params({"model": "logistic_regression", "C": 0.5, "features": len(FEATURES),
                       "train_years": "2007-2013"})
    for s in ["validation", "test"]:
        p = logreg.predict_proba(to_object(splits[s]))[:, 1]
        m = evaluate(splits[s]["is_default"], p, p)
        record("Logistic regression", s, m)
        mlflow.log_metrics({f"{s}_{k}": v for k, v in m.items()})
    # cloudpickle: the newer default (skops) rejects the numpy types inside this pipeline
    log_model(mlflow.sklearn, logreg, serialization_format="cloudpickle")
    logreg_run_id = run.info.run_id
print("Logged run", logreg_run_id)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model 2: LightGBM
# MAGIC LightGBM handles missing values and categories natively. Training stops early once the **2014
# MAGIC validation** score stops improving, which prevents overfitting without ever touching the 2015 test data.

# COMMAND ----------

params = dict(
    n_estimators=3000, learning_rate=0.03, num_leaves=31, min_child_samples=200,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
    random_state=42, verbose=-1,
)
lgbm = lgb.LGBMClassifier(**params)

with mlflow.start_run(run_name="lightgbm") as run:
    lgbm.fit(
        splits["train"][FEATURES], splits["train"]["is_default"],
        eval_set=[(splits["validation"][FEATURES], splits["validation"]["is_default"])],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    mlflow.log_params({**params, "model": "lightgbm", "best_iteration": lgbm.best_iteration_,
                       "features": len(FEATURES), "train_years": "2007-2013"})
    for s in ["validation", "test"]:
        p = lgbm.predict_proba(splits[s][FEATURES])[:, 1]
        m = evaluate(splits[s]["is_default"], p, p)
        record("LightGBM", s, m)
        mlflow.log_metrics({f"{s}_{k}": v for k, v in m.items()})
    log_model(mlflow.lightgbm, lgbm)
    lgbm_run_id = run.info.run_id
print(f"Logged run {lgbm_run_id}; best iteration {lgbm.best_iteration_}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Results: models vs. Lending Club

# COMMAND ----------

comparison = pd.DataFrame(results).sort_values(["split", "roc_auc"], ascending=[True, False])
spark.createDataFrame(comparison).write.mode("overwrite").option("overwriteSchema", "true") \
    .saveAsTable(T("gold_model_comparison"))
display(comparison)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature importance (LightGBM, by total gain)

# COMMAND ----------

importance = (
    pd.DataFrame({"feature": FEATURES,
                  "gain": lgbm.booster_.feature_importance(importance_type="gain")})
    .assign(share_pct=lambda d: (100 * d["gain"] / d["gain"].sum()).round(2))
    .sort_values("gain", ascending=False)
    .reset_index(drop=True)
)
spark.createDataFrame(importance).write.mode("overwrite").option("overwriteSchema", "true") \
    .saveAsTable(T("gold_feature_importance"))
display(importance)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save predictions for Phase 6 (risk grades, approval policy, expected loss)

# COMMAND ----------

preds = []
for s in ["train", "validation", "test"]:
    d = splits[s]
    preds.append(pd.DataFrame({
        "loan_id": d["loan_id"].astype("int64"),
        "split": s,
        "issue_year": d["issue_year"].astype(int),
        "is_default": d["is_default"],
        "grade": d["grade"].astype(str),
        "sub_grade": d["sub_grade"].astype(str),
        "int_rate": d["int_rate"].astype(float),
        "loan_amnt": d["loan_amnt"].astype(float),
        "pd_logreg": logreg.predict_proba(to_object(d))[:, 1],
        "pd_lgbm": lgbm.predict_proba(d[FEATURES])[:, 1],
    }))
spark.createDataFrame(pd.concat(preds, ignore_index=True)).write.mode("overwrite") \
    .option("overwriteSchema", "true").saveAsTable(T("gold_predictions"))
print(f"{T('gold_predictions')}: {spark.table(T('gold_predictions')).count():,} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register the LightGBM model in Unity Catalog (optional)
# MAGIC Registration makes the model versioned and reusable for scoring. If the workspace doesn't allow it,
# MAGIC the model is still saved in the MLflow run above.

# COMMAND ----------

try:
    mlflow.set_registry_uri("databricks-uc")
    version = mlflow.register_model(f"runs:/{lgbm_run_id}/model", T("default_risk_lgbm"))
    print(f"Registered {T('default_risk_lgbm')} version {version.version}")
except Exception as e:
    print(f"Registration skipped: {type(e).__name__}: {str(e)[:300]}")