# Databricks notebook source
# MAGIC %md
# MAGIC # 08 Export results
# MAGIC Writes the small summary tables to CSV in a volume, so they can be copied into the repo's `results/`
# MAGIC folder. The repo then shows the actual numbers without anyone needing a Databricks workspace.
# MAGIC Only aggregated results are exported; no loan-level data leaves the workspace.

# COMMAND ----------

import os

CATALOG = "workspace"
SCHEMA = "lending_club"
T = lambda name: f"{CATALOG}.{SCHEMA}.{name}"
OUT = f"/Volumes/{CATALOG}/{SCHEMA}/raw/exports"
os.makedirs(OUT, exist_ok=True)

SUMMARY_TABLES = [
    "gold_model_comparison", "gold_feature_importance", "gold_calibration", "gold_risk_bands",
    "gold_loan_selection", "gold_expected_loss", "gold_shap_importance", "gold_reason_codes",
    "silver_dq_log",
]
for name in SUMMARY_TABLES:
    spark.table(T(name)).toPandas().to_csv(f"{OUT}/{name}.csv", index=False)
    print("wrote", name)

# COMMAND ----------

# Row counts for every table in the schema, plus the modeling split sizes and default rates.
tables = [r.tableName for r in spark.sql(f"SHOW TABLES IN {CATALOG}.{SCHEMA}").collect()]
counts = spark.createDataFrame([(t, spark.table(T(t)).count()) for t in sorted(tables)], "table_name string, row_count long")
counts.toPandas().to_csv(f"{OUT}/table_counts.csv", index=False)
display(counts)

splits = spark.sql(f"""
    SELECT split, MIN(issue_year) AS first_year, MAX(issue_year) AS last_year,
           COUNT(*) AS loans, ROUND(100 * AVG(is_default), 2) AS default_rate_pct
    FROM {T('gold_features')} GROUP BY split ORDER BY first_year
""")
splits.toPandas().to_csv(f"{OUT}/split_summary.csv", index=False)
display(splits)

grades = spark.sql(f"""
    SELECT grade, COUNT(*) AS loans, ROUND(100 * AVG(is_default), 1) AS actual_default_pct,
           ROUND(AVG(int_rate), 2) AS avg_interest_rate
    FROM {T('gold_predictions')} WHERE split = 'test' GROUP BY grade ORDER BY grade
""")
grades.toPandas().to_csv(f"{OUT}/test_grades.csv", index=False)
print("Done:", sorted(os.listdir(OUT)))