# Databricks notebook source
# MAGIC %md
# MAGIC # 06 Evaluate: from a model to lending decisions
# MAGIC
# MAGIC Uses the 2015 **test** loans (never seen during training or tuning) to answer the questions a lender asks:
# MAGIC 1. **Calibration**: when the model predicts a 20% default chance, do about 20% default?
# MAGIC 2. **Risk bands**: do five model-based bands separate risk as cleanly as Lending Club's grades?
# MAGIC 3. **Loan selection**: if an investor funds only part of the loans, does choosing by model score beat
# MAGIC    choosing by Lending Club grade, at the same funding rate?
# MAGIC 4. **Expected loss**: PD × LGD × EAD, compared with the losses that actually happened.
# MAGIC 5. **Explanations (SHAP)**: what really drives the model, and the top reasons behind individual scores.
# MAGIC
# MAGIC Run this notebook **in the browser** (MLflow is used to load the trained model).

# COMMAND ----------

# MAGIC %pip install --quiet lightgbm skops

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os

os.environ["MLFLOW_TRACKING_URI"] = "databricks"
os.environ["MLFLOW_REGISTRY_URI"] = "databricks-uc"
import mlflow

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

import numpy as np
import pandas as pd

CATALOG = "workspace"
SCHEMA = "lending_club"
T = lambda name: f"{CATALOG}.{SCHEMA}.{name}"


def save(pdf, name):
    spark.createDataFrame(pdf).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(T(name))


preds = spark.table(T("gold_predictions")).toPandas()
outcomes = spark.table(T("silver_loan_outcomes")).select("loan_id", "funded_amnt", "total_pymnt").toPandas()
preds = preds.merge(outcomes, on="loan_id", how="left")
preds["net_return"] = preds["total_pymnt"] - preds["funded_amnt"]

test = preds[preds["split"] == "test"].copy()
valid = preds[preds["split"] == "validation"].copy()
train = preds[preds["split"] == "train"].copy()
print(f"Test loans: {len(test):,}   actual default rate: {100 * test['is_default'].mean():.2f}%")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Calibration: predicted vs. actual default rate, by score decile
# MAGIC The model was trained on 2007–2013, when defaults were lower than in 2015, so some under-prediction is
# MAGIC expected. Ranking (AUC) can be good even when the level is off; in practice the level is recalibrated
# MAGIC on recent data before use.

# COMMAND ----------

test["decile"] = pd.qcut(test["pd_lgbm"].rank(method="first"), 10, labels=range(1, 11)).astype(int)
calibration = (
    test.groupby("decile")
    .agg(loans=("loan_id", "size"),
         predicted_pd_pct=("pd_lgbm", lambda x: round(100 * x.mean(), 2)),
         actual_default_pct=("is_default", lambda x: round(100 * x.mean(), 2)))
    .reset_index()
)
save(calibration, "gold_calibration")
display(calibration)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Model risk bands vs. Lending Club grades
# MAGIC Band cut-offs are set on the **2014 validation** scores (equal-sized fifths), then applied to 2015, so the
# MAGIC bands are defined without looking at the test year.

# COMMAND ----------

cuts = np.quantile(valid["pd_lgbm"], [0.2, 0.4, 0.6, 0.8])
band_labels = ["1 (lowest risk)", "2", "3", "4", "5 (highest risk)"]
test["risk_band"] = pd.cut(test["pd_lgbm"], [-np.inf, *cuts, np.inf], labels=band_labels).astype(str)

bands = (
    test.groupby("risk_band")
    .agg(loans=("loan_id", "size"),
         avg_predicted_pd_pct=("pd_lgbm", lambda x: round(100 * x.mean(), 1)),
         actual_default_pct=("is_default", lambda x: round(100 * x.mean(), 1)),
         avg_interest_rate=("int_rate", lambda x: round(x.mean(), 2)),
         net_return_pct=("net_return", "sum"),
         funded=("funded_amnt", "sum"))
    .reset_index()
)
bands["net_return_pct"] = (100 * bands["net_return_pct"] / bands["funded"]).round(1)
bands = bands.drop(columns="funded")
save(bands, "gold_risk_bands")
display(bands)

# COMMAND ----------

grades = (
    test.groupby("grade")
    .agg(loans=("loan_id", "size"),
         actual_default_pct=("is_default", lambda x: round(100 * x.mean(), 1)),
         avg_interest_rate=("int_rate", lambda x: round(x.mean(), 2)))
    .reset_index()
)
display(grades)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Loan selection: model score vs. Lending Club grade
# MAGIC Every 2015 loan here was already approved by Lending Club, so this is the **investor's** question: if you
# MAGIC fund only the safest X% of loans, which ranking picks a better portfolio?
# MAGIC
# MAGIC - **Model**: fund the loans with the lowest predicted default probability.
# MAGIC - **Lending Club**: fund the best sub-grades first (A1, A2, ...).
# MAGIC
# MAGIC Net return = total repaid ÷ amount funded − 1, over the life of the loans (not annualized).

# COMMAND ----------

SUB_GRADES = [f"{g}{n}" for g in "ABCDEFG" for n in range(1, 6)]
test["sub_grade_rank"] = test["sub_grade"].map({sg: i for i, sg in enumerate(SUB_GRADES)})
# Small random tie-breaker so loans in the same sub-grade are picked in arbitrary order, not by ID.
rng = np.random.default_rng(42)
test["tiebreak"] = rng.random(len(test))


def portfolio(order_cols, share):
    chosen = test.sort_values(order_cols).head(int(len(test) * share))
    return (100 * chosen["is_default"].mean(),
            100 * chosen["net_return"].sum() / chosen["funded_amnt"].sum())


rows = []
for share in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    m_def, m_ret = portfolio(["pd_lgbm"], share)
    g_def, g_ret = portfolio(["sub_grade_rank", "tiebreak"], share)
    rows.append({"funded_share_pct": int(share * 100),
                 "model_default_pct": round(m_def, 2), "grade_default_pct": round(g_def, 2),
                 "model_net_return_pct": round(m_ret, 2), "grade_net_return_pct": round(g_ret, 2)})
selection = pd.DataFrame(rows)
selection["return_advantage_pts"] = (selection["model_net_return_pct"] - selection["grade_net_return_pct"]).round(2)
save(selection, "gold_loan_selection")
display(selection)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Expected loss: PD × LGD × EAD
# MAGIC - **PD**: the model's predicted default probability.
# MAGIC - **LGD** (loss given default): the share of the funded amount lost when a loan defaults, estimated from
# MAGIC   **2007–2013** defaults only, using the outcome table. This is the legitimate use of outcome data.
# MAGIC - **EAD** (exposure at default): approximated by the funded amount.

# COMMAND ----------

train_defaults = train[(train["is_default"] == 1) & (train["funded_amnt"] > 0)]
lgd = float(((train_defaults["funded_amnt"] - train_defaults["total_pymnt"]).clip(lower=0)
             / train_defaults["funded_amnt"]).mean())

test["expected_loss"] = test["pd_lgbm"] * lgd * test["funded_amnt"]
actual_loss = (test["funded_amnt"] - test["total_pymnt"]).clip(lower=0).where(test["is_default"] == 1, 0)

expected_loss = pd.DataFrame([{
    "lgd_pct": round(100 * lgd, 1),
    "funded_total": round(test["funded_amnt"].sum()),
    "expected_loss": round(test["expected_loss"].sum()),
    "actual_loss": round(actual_loss.sum()),
    "expected_loss_pct_of_funded": round(100 * test["expected_loss"].sum() / test["funded_amnt"].sum(), 2),
    "actual_loss_pct_of_funded": round(100 * actual_loss.sum() / test["funded_amnt"].sum(), 2),
}])
save(expected_loss, "gold_expected_loss")
display(expected_loss)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Explanations with SHAP
# MAGIC Gain-based importance (from training) is known to overstate features with many categories, such as
# MAGIC `addr_state` (50 states). SHAP values measure how much each feature actually moves each prediction, so they
# MAGIC give a fairer picture. LightGBM computes them directly (`pred_contrib=True`), so no extra library is needed.

# COMMAND ----------

user = spark.sql("SELECT current_user()").first()[0]
runs = mlflow.search_runs(
    experiment_names=[f"/Users/{user}/lending-club-default-risk"],
    filter_string="tags.mlflow.runName = 'lightgbm'",
    order_by=["start_time DESC"],
)
lgbm_run_id = runs.iloc[0]["run_id"]
model = mlflow.lightgbm.load_model(f"runs:/{lgbm_run_id}/model")
print("Loaded LightGBM from run", lgbm_run_id)

# COMMAND ----------

# Rebuild the test features exactly as in 05_train_default_models (category levels from training data only).
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

feat = spark.table(T("gold_features")).toPandas()
for c in NUMERIC_FEATURES + FLAG_FEATURES:
    feat[c] = pd.to_numeric(feat[c], errors="coerce").astype("float64")
is_train = feat["split"] == "train"
for c in CATEGORICAL_FEATURES:
    feat[c] = pd.Categorical(feat[c], categories=sorted(feat.loc[is_train, c].dropna().unique()))

sample = feat[feat["split"] == "test"].sample(n=20000, random_state=42).reset_index(drop=True)
contrib = model.predict(sample[FEATURES], pred_contrib=True)   # last column is the baseline
shap_values = pd.DataFrame(contrib[:, :-1], columns=FEATURES)

shap_importance = (
    shap_values.abs().mean().rename("mean_abs_shap").reset_index().rename(columns={"index": "feature"})
    .assign(share_pct=lambda d: (100 * d["mean_abs_shap"] / d["mean_abs_shap"].sum()).round(2))
    .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
)
gain = spark.table(T("gold_feature_importance")).toPandas()[["feature", "share_pct"]] \
    .rename(columns={"share_pct": "gain_share_pct"})
shap_importance = shap_importance.merge(gain, on="feature", how="left")
save(shap_importance, "gold_shap_importance")
display(shap_importance)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Reason codes for individual loans
# MAGIC Lenders must be able to tell rejected applicants why. For three example loans (low, medium, and high
# MAGIC predicted risk), these are the three features that pushed each score up the most.

# COMMAND ----------

sample["pd"] = model.predict_proba(sample[FEATURES])[:, 1]
picks = {
    "low risk": sample["pd"].idxmin(),
    "medium risk": (sample["pd"] - sample["pd"].median()).abs().idxmin(),
    "high risk": sample["pd"].idxmax(),
}

reasons = []
for label, i in picks.items():
    top = shap_values.loc[i].sort_values(ascending=False).head(3)
    for rank, (feature, value) in enumerate(top.items(), start=1):
        reasons.append({
            "example": label,
            "loan_id": int(sample.loc[i, "loan_id"]),
            "predicted_pd_pct": round(100 * sample.loc[i, "pd"], 1),
            "rank": rank,
            "feature": feature,
            "feature_value": str(sample.loc[i, feature]),
            "shap_log_odds": round(float(value), 3),
        })
reason_codes = pd.DataFrame(reasons)
save(reason_codes, "gold_reason_codes")
display(reason_codes)