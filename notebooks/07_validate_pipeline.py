# Databricks notebook source
# MAGIC %md
# MAGIC # 07 Quality checks
# MAGIC The last task in the job. Every check below must pass; if any fails, this notebook raises an error, the
# MAGIC job fails, and nothing downstream (like the dashboard) is silently fed bad data.
# MAGIC
# MAGIC Checks cover **data integrity** (silver), **modeling design** (cohort, time split, leakage) and **model
# MAGIC outputs** (predictions and a plausible test AUC). An AUC that is suspiciously *high* fails too, because in
# MAGIC credit risk that almost always means outcome information leaked into the features.

# COMMAND ----------

from pyspark.sql import functions as F

CATALOG = "workspace"
SCHEMA = "lending_club"
T = lambda name: f"{CATALOG}.{SCHEMA}.{name}"

results = []


def check(name, passed, detail=""):
    results.append((name, bool(passed), detail))


def table_exists(name):
    return spark.catalog.tableExists(T(name))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Silver: data integrity

# COMMAND ----------

loans = spark.table(T("silver_loans"))
n_loans = loans.count()
check("silver_loans has the full dataset (> 2.2M loans)", n_loans > 2_200_000, f"{n_loans:,} loans")
check("loan_id is unique", loans.select("loan_id").distinct().count() == n_loans)
check("no loan is missing its issue date", loans.filter("issue_date IS NULL").count() == 0)
check("is_default is only 0, 1, or empty", loans.filter("is_default NOT IN (0, 1)").count() == 0)
check("only resolved loans have a target",
      loans.filter("(is_resolved AND is_default IS NULL) OR (NOT is_resolved AND is_default IS NOT NULL)").count() == 0)

outcome_columns = {"total_pymnt", "recoveries", "total_rec_prncp", "last_pymnt_amnt", "last_fico_range_high"}
leaked = outcome_columns & set(loans.columns)
check("no post-issue outcome columns in silver_loans", not leaked, f"found: {sorted(leaked)}" if leaked else "")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Gold features: cohort, time split, and leakage

# COMMAND ----------

gold = spark.table(T("gold_features"))
n_gold = gold.count()

cohort = loans.filter("term_months = 36 AND issue_year BETWEEN 2007 AND 2015 AND is_resolved").count()
check("gold_features matches the modeling cohort (36-month, 2007-2015, resolved)", n_gold == cohort,
      f"{n_gold:,} vs {cohort:,}")
check("every gold row has a target", gold.filter("is_default IS NULL").count() == 0)

years = {r["split"]: (r["lo"], r["hi"]) for r in
         gold.groupBy("split").agg(F.min("issue_year").alias("lo"), F.max("issue_year").alias("hi")).collect()}
check("time split is in order: train < validation < test",
      years["train"][1] < years["validation"][0] <= years["validation"][1] < years["test"][0],
      str(years))

forbidden_in_gold = {"installment", "payment_to_income", "loan_status", "funded_amnt", "total_pymnt",
                     "recoveries", "last_pymnt_amnt"}
present = forbidden_in_gold & set(gold.columns)
check("gold_features contains no leakage columns", not present, f"found: {sorted(present)}" if present else "")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model outputs

# COMMAND ----------

if table_exists("gold_predictions") and table_exists("gold_model_comparison"):
    preds = spark.table(T("gold_predictions"))
    check("one prediction per modeling loan", preds.count() == n_gold, f"{preds.count():,} vs {n_gold:,}")
    check("predicted probabilities are between 0 and 1",
          preds.filter("pd_lgbm < 0 OR pd_lgbm > 1 OR pd_logreg < 0 OR pd_logreg > 1").count() == 0)

    comp = {(r["model"], r["split"]): r["roc_auc"] for r in spark.table(T("gold_model_comparison")).collect()}
    test_auc = comp.get(("LightGBM", "test"))
    grade_auc = comp.get(("Lending Club sub-grade", "test"))
    check("LightGBM test AUC is plausible (0.60-0.80; higher suggests leakage)",
          test_auc is not None and 0.60 <= test_auc <= 0.80, f"AUC {test_auc}")
    check("LightGBM is no worse than Lending Club's grades by more than 0.01 AUC",
          test_auc is not None and grade_auc is not None and test_auc >= grade_auc - 0.01,
          f"model {test_auc} vs grades {grade_auc}")
else:
    print("Model tables not found yet; skipping model checks (run 05_train_default_models first).")

# COMMAND ----------

for name, passed, detail in results:
    print(f"{'PASS' if passed else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))

failed = [name for name, passed, _ in results if not passed]
if failed:
    raise Exception(f"{len(failed)} quality check(s) failed: {failed}")
print(f"\nAll {len(results)} checks passed.")