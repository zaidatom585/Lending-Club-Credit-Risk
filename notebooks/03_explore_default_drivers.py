# Databricks notebook source
# MAGIC %md
# MAGIC # 03 Exploration
# MAGIC Run this notebook **cell by cell in the browser** and note the findings.
# MAGIC Default rates use **resolved loans only** (fully paid, charged off, or defaulted), since loans still in
# MAGIC progress don't have a final outcome yet.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Automatic check: did silver's conversions work?
# MAGIC Silver uses safe conversions that turn unreadable values into NULL instead of failing. If a whole column
# MAGIC were in an unexpected format, it would silently become empty, so this cell fails loudly if any key column
# MAGIC is more than 20% empty.

# COMMAND ----------

from pyspark.sql import functions as F

loans = spark.table("workspace.lending_club.silver_loans")
total = loans.count()
key_columns = ["issue_date", "term_months", "loan_amnt", "int_rate", "installment", "annual_inc",
               "fico_avg", "grade", "purpose", "addr_state", "credit_history_years"]
null_counts = loans.agg(*[F.sum(F.col(c).isNull().cast("int")).alias(c) for c in key_columns]).first().asDict()

problems = []
for c in key_columns:
    pct = 100 * null_counts[c] / total
    print(f"{c:<22} {null_counts[c]:>9,} empty ({pct:5.2f}%)")
    if pct > 20:
        problems.append(c)
if problems:
    raise ValueError(f"These columns are mostly empty, so a conversion in silver likely failed: {problems}")
print(f"\nAll key columns look healthy across {total:,} loans.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Loan status and the overall default rate

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT loan_status, is_default, COUNT(*) AS loans,
# MAGIC        ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct_of_all
# MAGIC FROM workspace.lending_club.silver_loans
# MAGIC GROUP BY loan_status, is_default
# MAGIC ORDER BY loans DESC

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Vintage analysis: outcomes by issue year and term
# MAGIC Recent years have many unresolved loans, and the ones already resolved are biased toward early defaults.
# MAGIC This table decides which years are safe to model on.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT issue_year, term_months,
# MAGIC        COUNT(*)                                                         AS loans,
# MAGIC        ROUND(100.0 * AVG(CASE WHEN is_resolved THEN 1 ELSE 0 END), 1)   AS pct_resolved,
# MAGIC        ROUND(100.0 * AVG(CASE WHEN is_resolved THEN is_default END), 1) AS default_rate_pct
# MAGIC FROM workspace.lending_club.silver_loans
# MAGIC GROUP BY issue_year, term_months
# MAGIC ORDER BY issue_year, term_months

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Lending Club's grades: does risk rise from A to G?
# MAGIC This is the benchmark our model must beat.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT grade,
# MAGIC        COUNT(*)                        AS resolved_loans,
# MAGIC        ROUND(AVG(int_rate), 2)         AS avg_interest_rate,
# MAGIC        ROUND(100.0 * AVG(is_default), 1) AS default_rate_pct
# MAGIC FROM workspace.lending_club.silver_loans
# MAGIC WHERE is_resolved
# MAGIC GROUP BY grade
# MAGIC ORDER BY grade

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Borrower factors

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Credit score bands
# MAGIC SELECT CAST(FLOOR(fico_avg / 20) * 20 AS INT) AS fico_band,
# MAGIC        COUNT(*) AS resolved_loans,
# MAGIC        ROUND(100.0 * AVG(is_default), 1) AS default_rate_pct
# MAGIC FROM workspace.lending_club.silver_loans
# MAGIC WHERE is_resolved AND fico_avg IS NOT NULL
# MAGIC GROUP BY 1
# MAGIC ORDER BY 1

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Debt-to-income bands
# MAGIC SELECT CASE WHEN dti < 10 THEN '1. under 10'
# MAGIC             WHEN dti < 20 THEN '2. 10-20'
# MAGIC             WHEN dti < 30 THEN '3. 20-30'
# MAGIC             WHEN dti < 40 THEN '4. 30-40'
# MAGIC             ELSE '5. 40+' END AS dti_band,
# MAGIC        COUNT(*) AS resolved_loans,
# MAGIC        ROUND(100.0 * AVG(is_default), 1) AS default_rate_pct
# MAGIC FROM workspace.lending_club.silver_loans
# MAGIC WHERE is_resolved AND dti IS NOT NULL AND dti >= 0
# MAGIC GROUP BY 1
# MAGIC ORDER BY 1

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Loan size relative to income
# MAGIC SELECT CASE WHEN loan_to_income < 0.1 THEN '1. under 10%'
# MAGIC             WHEN loan_to_income < 0.2 THEN '2. 10-20%'
# MAGIC             WHEN loan_to_income < 0.3 THEN '3. 20-30%'
# MAGIC             WHEN loan_to_income < 0.5 THEN '4. 30-50%'
# MAGIC             ELSE '5. 50%+' END AS loan_to_income_band,
# MAGIC        COUNT(*) AS resolved_loans,
# MAGIC        ROUND(100.0 * AVG(is_default), 1) AS default_rate_pct
# MAGIC FROM workspace.lending_club.silver_loans
# MAGIC WHERE is_resolved AND loan_to_income IS NOT NULL
# MAGIC GROUP BY 1
# MAGIC ORDER BY 1

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Loan purpose (largest categories)
# MAGIC SELECT purpose, COUNT(*) AS resolved_loans,
# MAGIC        ROUND(100.0 * AVG(is_default), 1) AS default_rate_pct
# MAGIC FROM workspace.lending_club.silver_loans
# MAGIC WHERE is_resolved
# MAGIC GROUP BY purpose
# MAGIC HAVING COUNT(*) >= 1000
# MAGIC ORDER BY default_rate_pct DESC

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Home ownership and employment length
# MAGIC SELECT home_ownership, COUNT(*) AS resolved_loans,
# MAGIC        ROUND(100.0 * AVG(is_default), 1) AS default_rate_pct
# MAGIC FROM workspace.lending_club.silver_loans
# MAGIC WHERE is_resolved
# MAGIC GROUP BY home_ownership
# MAGIC HAVING COUNT(*) >= 1000
# MAGIC ORDER BY default_rate_pct DESC

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT emp_length_years, COUNT(*) AS resolved_loans,
# MAGIC        ROUND(100.0 * AVG(is_default), 1) AS default_rate_pct
# MAGIC FROM workspace.lending_club.silver_loans
# MAGIC WHERE is_resolved
# MAGIC GROUP BY emp_length_years
# MAGIC ORDER BY emp_length_years

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Does a higher interest rate compensate for higher risk?
# MAGIC Compares, by grade, the interest earned against the share of loans that default. If default rates rise
# MAGIC faster than interest rates, the riskiest grades may be under-priced.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT l.grade,
# MAGIC        COUNT(*)                                          AS resolved_loans,
# MAGIC        ROUND(AVG(l.int_rate), 2)                         AS avg_interest_rate,
# MAGIC        ROUND(100.0 * AVG(l.is_default), 1)               AS default_rate_pct,
# MAGIC        ROUND(100.0 * (SUM(o.total_pymnt) - SUM(o.funded_amnt)) / SUM(o.funded_amnt), 1) AS realized_return_pct
# MAGIC FROM workspace.lending_club.silver_loans l
# MAGIC JOIN workspace.lending_club.silver_loan_outcomes o USING (loan_id)
# MAGIC WHERE l.is_resolved
# MAGIC GROUP BY l.grade
# MAGIC ORDER BY l.grade

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Accepted vs. rejected applications

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Approval rate by year
# MAGIC WITH acc AS (SELECT issue_year AS yr, COUNT(*) AS accepted FROM workspace.lending_club.silver_loans GROUP BY 1),
# MAGIC      rej AS (SELECT application_year AS yr, COUNT(*) AS rejected FROM workspace.lending_club.silver_rejected GROUP BY 1)
# MAGIC SELECT acc.yr AS year, accepted, rejected,
# MAGIC        ROUND(100.0 * accepted / (accepted + rejected), 1) AS approval_rate_pct
# MAGIC FROM acc JOIN rej USING (yr)
# MAGIC ORDER BY year

# COMMAND ----------

# MAGIC %sql
# MAGIC -- How rejected applicants differed (medians)
# MAGIC SELECT 'accepted' AS group_name, COUNT(*) AS applications,
# MAGIC        PERCENTILE_APPROX(loan_amnt, 0.5) AS median_amount, PERCENTILE_APPROX(dti, 0.5) AS median_dti
# MAGIC FROM workspace.lending_club.silver_loans
# MAGIC UNION ALL
# MAGIC SELECT 'rejected', COUNT(*),
# MAGIC        PERCENTILE_APPROX(amount_requested, 0.5), PERCENTILE_APPROX(dti, 0.5)
# MAGIC FROM workspace.lending_club.silver_rejected
# MAGIC WHERE dti >= 0