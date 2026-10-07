# Databricks notebook source
# MAGIC %md
# MAGIC # 04 Features (gold)
# MAGIC Builds one row per loan for modeling, using **application-time information only**.
# MAGIC
# MAGIC **Which loans:** 36-month loans issued 2007–2015. Exploration showed these are 99.9–100% resolved, so
# MAGIC their outcomes are complete. Later 60-month loans are only partly resolved, and their early defaults
# MAGIC would bias the model.
# MAGIC
# MAGIC **Time-based split** (how a lender would actually deploy a model):
# MAGIC - `train`: issued 2007–2013
# MAGIC - `validation`: issued 2014 (for tuning)
# MAGIC - `test`: issued 2015 (final, untouched evaluation)
# MAGIC
# MAGIC **Excluded from features:** `grade`, `sub_grade`, and `int_rate` are Lending Club's own risk assessment,
# MAGIC kept only as a benchmark. `installment` is excluded too: for a fixed 36-month term it is calculated from
# MAGIC the interest rate, so it would quietly leak Lending Club's assessment into the model.

# COMMAND ----------

from pyspark.sql import functions as F

CATALOG = "workspace"
SCHEMA = "lending_club"
T = lambda name: f"{CATALOG}.{SCHEMA}.{name}"

loans = spark.table(T("silver_loans"))
cohort = loans.filter(
    (F.col("term_months") == 36)
    & F.col("issue_year").between(2007, 2015)
    & F.col("is_resolved")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature definitions

# COMMAND ----------

NUMERIC_FEATURES = [
    "loan_amnt", "annual_inc_log", "dti", "fico_avg", "inq_last_6mths", "delinq_2yrs",
    "open_acc", "total_acc", "pub_rec", "revol_bal_log", "revol_util", "mort_acc",
    "pub_rec_bankruptcies", "credit_history_years", "emp_length_years", "loan_to_income",
    "months_since_last_delinq", "tot_cur_bal_log", "total_rev_hi_lim_log",
    "acc_open_past_24mths", "bc_util", "num_actv_rev_tl", "percent_bc_gt_75",
]
FLAG_FEATURES = ["emp_length_missing", "never_delinquent"]
CATEGORICAL_FEATURES = ["home_ownership", "verification_status", "purpose", "addr_state", "application_type"]
BENCHMARK_COLUMNS = ["grade", "sub_grade", "int_rate"]

features = cohort.select(
    "loan_id", "issue_date", "issue_year",
    F.when(F.col("issue_year") <= 2013, "train")
     .when(F.col("issue_year") == 2014, "validation")
     .otherwise("test").alias("split"),
    "is_default",
    *BENCHMARK_COLUMNS,
    # Size and affordability
    "loan_amnt", "dti", "loan_to_income",
    F.log1p("annual_inc").alias("annual_inc_log"),
    # Credit profile
    "fico_avg", "inq_last_6mths", "delinq_2yrs", "open_acc", "total_acc", "pub_rec",
    F.log1p("revol_bal").alias("revol_bal_log"), "revol_util", "mort_acc",
    "pub_rec_bankruptcies", "credit_history_years",
    # Delinquency history: an empty value means "never delinquent", which is itself informative
    F.col("mths_since_last_delinq").alias("months_since_last_delinq"),
    F.col("mths_since_last_delinq").isNull().cast("int").alias("never_delinquent"),
    # Employment: missing length was the strongest employment signal in exploration (26.9% vs ~20% default)
    "emp_length_years",
    F.col("emp_length_years").isNull().cast("int").alias("emp_length_missing"),
    # Newer bureau fields (reported from about 2012 onward; empty for earlier loans)
    F.log1p("tot_cur_bal").alias("tot_cur_bal_log"),
    F.log1p("total_rev_hi_lim").alias("total_rev_hi_lim_log"),
    "acc_open_past_24mths", "bc_util", "num_actv_rev_tl", "percent_bc_gt_75",
    # Categories
    *CATEGORICAL_FEATURES,
)

(features.write.format("delta").mode("overwrite")
 .option("overwriteSchema", "true").saveAsTable(T("gold_features")))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary of the modeling dataset

# COMMAND ----------

gold = spark.table(T("gold_features"))
display(
    gold.groupBy("split")
    .agg(
        F.count("*").alias("loans"),
        F.min("issue_year").alias("from_year"),
        F.max("issue_year").alias("to_year"),
        F.round(100 * F.avg("is_default"), 2).alias("default_rate_pct"),
    )
    .orderBy("from_year")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Missing values per feature and split
# MAGIC Newer bureau fields are expected to be empty for older loans. The model must handle this.

# COMMAND ----------

display(
    gold.groupBy("split").agg(
        *[F.round(100 * F.avg(F.col(c).isNull().cast("int")), 1).alias(c)
          for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    ).orderBy("split")
)

# COMMAND ----------

# Leakage guard: these must never appear among the model's feature lists.
FORBIDDEN = {"grade", "sub_grade", "int_rate", "installment", "payment_to_income", "loan_status",
             "total_pymnt", "recoveries", "last_pymnt_amnt", "funded_amnt"}
model_features = set(NUMERIC_FEATURES + FLAG_FEATURES + CATEGORICAL_FEATURES)
assert not (model_features & FORBIDDEN), f"Leakage: {model_features & FORBIDDEN}"
print(f"{len(model_features)} model features, none of them forbidden.")