# Databricks notebook source
# MAGIC %md
# MAGIC # 02 Silver: typed, cleaned, and leakage-safe
# MAGIC
# MAGIC Turns the all-text bronze tables into properly typed silver tables.
# MAGIC
# MAGIC **Leakage protection.** About a third of Lending Club's columns describe what happened *after* a loan was
# MAGIC issued (payments received, recoveries, latest credit score). Silver keeps an explicit **allowlist** of
# MAGIC application-time columns in `silver_loans`, so outcome information can't reach the model by accident.
# MAGIC Outcome columns go to `silver_loan_outcomes`, used later only to estimate losses.
# MAGIC
# MAGIC **Safe conversions.** Serverless compute uses ANSI SQL, where a bad conversion or a division by zero
# MAGIC fails the job. `try_cast`, `try_to_timestamp`, and `try_divide` return NULL instead.

# COMMAND ----------

from pyspark.sql import functions as F

CATALOG = "workspace"
SCHEMA = "lending_club"
T = lambda name: f"{CATALOG}.{SCHEMA}.{name}"

bronze = spark.table(T("bronze_accepted"))
cols = set(bronze.columns)


def num(c):
    """Text to number, tolerating '%' signs, spaces, and empty strings."""
    return F.expr(f"try_cast(regexp_replace(trim(`{c}`), '%', '') AS DOUBLE)")


def month_date(c):
    """'Dec-2015' -> 2015-12-01."""
    return F.expr(f"to_date(try_to_timestamp(concat('01-', trim(`{c}`)), 'dd-MMM-yyyy'))")


def keep(names):
    """Only columns that exist in this file, so the notebook doesn't break on a slightly different version."""
    return [c for c in names if c in cols]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Remove invalid rows and duplicates, and log what was removed
# MAGIC The CSV contains summary lines such as "Total amount funded in policy code 1" mixed in with the loans.
# MAGIC These have no numeric loan ID.

# COMMAND ----------

valid_id = F.col("id").rlike("^[0-9]+$")
has_amount = num("loan_amnt").isNotNull()
has_issue_date = month_date("issue_d").isNotNull()

dq_counts = bronze.agg(
    F.count("*").alias("input_rows"),
    F.sum(F.when(~valid_id | F.col("id").isNull(), 1).otherwise(0)).alias("invalid_or_summary_rows"),
    F.sum(F.when(valid_id & ~has_amount, 1).otherwise(0)).alias("missing_loan_amount"),
    F.sum(F.when(valid_id & has_amount & ~has_issue_date, 1).otherwise(0)).alias("missing_issue_date"),
).first().asDict()

valid = bronze.filter(valid_id & has_amount & has_issue_date)
valid_rows = valid.count()
deduped = valid.dropDuplicates(["id"])
deduped_rows = deduped.count()
dq_counts["duplicate_ids_removed"] = valid_rows - deduped_rows
dq_counts["rows_kept"] = deduped_rows
print(dq_counts)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. The target: did the loan default?
# MAGIC - **0** = fully paid
# MAGIC - **1** = charged off or defaulted
# MAGIC - **NULL** = still in progress (current, late, in grace period): the outcome isn't known yet,
# MAGIC   so these loans are kept for reporting but excluded from modeling.

# COMMAND ----------

status = F.trim(F.col("loan_status"))
target = (
    F.when(status.isin("Fully Paid", "Does not meet the credit policy. Status:Fully Paid"), 0)
    .when(status.isin("Charged Off", "Default", "Does not meet the credit policy. Status:Charged Off"), 1)
)

display(deduped.groupBy(status.alias("loan_status")).count().orderBy(F.desc("count")))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. `silver_loans`: application-time columns only
# MAGIC `grade`, `sub_grade`, and `int_rate` are Lending Club's own risk assessment at origination. They are kept
# MAGIC as a **benchmark** to compare our model against, and excluded from the model's features later.

# COMMAND ----------

NUMERIC_APP_COLUMNS = keep([
    "loan_amnt", "int_rate", "installment", "annual_inc", "dti",
    "delinq_2yrs", "fico_range_low", "fico_range_high", "inq_last_6mths",
    "mths_since_last_delinq", "mths_since_last_record", "open_acc", "pub_rec",
    "revol_bal", "revol_util", "total_acc", "mort_acc", "pub_rec_bankruptcies",
    "tot_cur_bal", "total_rev_hi_lim", "acc_open_past_24mths", "bc_util",
    "num_actv_rev_tl", "percent_bc_gt_75", "tax_liens",
])
TEXT_APP_COLUMNS = keep([
    "grade", "sub_grade", "home_ownership", "verification_status", "purpose",
    "addr_state", "initial_list_status", "application_type",
])

emp_length = F.trim(F.col("emp_length"))

silver_loans = deduped.select(
    F.col("id").cast("bigint").alias("loan_id"),
    month_date("issue_d").alias("issue_date"),
    F.expr("try_cast(regexp_extract(term, '([0-9]+)', 1) AS INT)").alias("term_months"),
    *[num(c).alias(c) for c in NUMERIC_APP_COLUMNS],
    *[F.trim(F.col(c)).alias(c) for c in TEXT_APP_COLUMNS],
    F.when(emp_length.isNull() | emp_length.isin("n/a", ""), None)
     .when(emp_length.contains("<"), 0)
     .otherwise(F.expr("try_cast(regexp_extract(emp_length, '([0-9]+)', 1) AS INT)"))
     .alias("emp_length_years"),
    month_date("earliest_cr_line").alias("earliest_credit_line"),
    status.alias("loan_status"),
    target.alias("is_default"),
)

silver_loans = (
    silver_loans
    .withColumn("issue_year", F.year("issue_date"))
    .withColumn("fico_avg", (F.col("fico_range_low") + F.col("fico_range_high")) / 2)
    .withColumn("credit_history_years",
                F.round(F.months_between("issue_date", "earliest_credit_line") / 12, 1))
    .withColumn("loan_to_income", F.expr("round(try_divide(loan_amnt, annual_inc), 4)"))
    .withColumn("payment_to_income", F.expr("round(try_divide(installment * 12, annual_inc), 4)"))
    .withColumn("is_resolved", F.col("is_default").isNotNull())
)

(silver_loans.write.format("delta").mode("overwrite")
 .option("overwriteSchema", "true").saveAsTable(T("silver_loans")))
print(f"{T('silver_loans')}: {spark.table(T('silver_loans')).count():,} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. `silver_loan_outcomes`: what happened after issue (never used as model features)

# COMMAND ----------

OUTCOME_NUMERIC = keep([
    "funded_amnt", "total_pymnt", "total_rec_prncp", "total_rec_int", "total_rec_late_fee",
    "recoveries", "collection_recovery_fee", "last_pymnt_amnt", "out_prncp",
    "last_fico_range_low", "last_fico_range_high",
])
outcomes = deduped.select(
    F.col("id").cast("bigint").alias("loan_id"),
    *[num(c).alias(c) for c in OUTCOME_NUMERIC],
    *([month_date("last_pymnt_d").alias("last_payment_date")] if "last_pymnt_d" in cols else []),
    status.alias("loan_status"),
)
(outcomes.write.format("delta").mode("overwrite")
 .option("overwriteSchema", "true").saveAsTable(T("silver_loan_outcomes")))
print(f"{T('silver_loan_outcomes')}: {spark.table(T('silver_loan_outcomes')).count():,} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. `silver_rejected`: rejected applications

# COMMAND ----------

rejected = spark.table(T("bronze_rejected"))
silver_rejected = rejected.select(
    F.expr("try_cast(trim(amount_requested) AS DOUBLE)").alias("amount_requested"),
    F.expr("to_date(try_to_timestamp(trim(application_date), 'yyyy-MM-dd'))").alias("application_date"),
    F.trim("loan_title").alias("loan_title"),
    F.expr("try_cast(trim(risk_score) AS DOUBLE)").alias("risk_score"),
    F.expr("try_cast(regexp_replace(trim(debt_to_income_ratio), '%', '') AS DOUBLE)").alias("dti"),
    F.trim("state").alias("state"),
    F.trim("employment_length").alias("employment_length"),
).withColumn("application_year", F.year("application_date"))

(silver_rejected.write.format("delta").mode("overwrite")
 .option("overwriteSchema", "true").saveAsTable(T("silver_rejected")))
print(f"{T('silver_rejected')}: {spark.table(T('silver_rejected')).count():,} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Data quality log

# COMMAND ----------

loans = spark.table(T("silver_loans"))
dq_counts["resolved_loans"] = loans.filter("is_resolved").count()
dq_counts["unresolved_loans"] = loans.filter("NOT is_resolved").count()
dq_counts["defaults"] = loans.filter("is_default = 1").count()

dq = spark.createDataFrame([dq_counts]).withColumn("run_at", F.current_timestamp())
dq.write.format("delta").mode("append").option("mergeSchema", "true").saveAsTable(T("silver_dq_log"))
display(dq)