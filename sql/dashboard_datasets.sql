-- =====================================================================
-- Lending Club credit risk dashboard: one query per dataset.
-- In the dashboard editor: Data tab -> "Create from SQL" -> paste ONE query,
-- then rename the dataset to the name in its header (e.g. ds_headline).
-- All numbers are on the 2015 test year unless stated otherwise.
-- =====================================================================


-- ds_headline  (4 counter widgets)
SELECT
  MAX(CASE WHEN model = 'LightGBM'               THEN roc_auc END) AS model_auc,
  MAX(CASE WHEN model = 'Lending Club sub-grade' THEN roc_auc END) AS grade_auc,
  MAX(CASE WHEN model = 'LightGBM'               THEN ks      END) AS model_ks,
  (SELECT COUNT(*) FROM workspace.lending_club.gold_predictions WHERE split = 'test') AS test_loans
FROM workspace.lending_club.gold_model_comparison
WHERE split = 'test';


-- ds_model_comparison  (table widget)
SELECT model,
       ROUND(roc_auc, 4) AS auc,
       ROUND(gini, 4)    AS gini,
       ROUND(ks, 4)      AS ks,
       ROUND(pr_auc, 4)  AS pr_auc,
       ROUND(brier, 4)   AS brier
FROM workspace.lending_club.gold_model_comparison
WHERE split = 'test'
ORDER BY roc_auc DESC;


-- ds_loan_selection  (line chart: X funded_share_pct, Y both return columns)
SELECT funded_share_pct,
       model_net_return_pct AS `Model (LightGBM)`,
       grade_net_return_pct AS `Lending Club sub-grade`,
       return_advantage_pts
FROM workspace.lending_club.gold_loan_selection
ORDER BY funded_share_pct;


-- ds_calibration  (bar chart: X decile, Y predicted + actual, grouped)
SELECT decile,
       predicted_pd_pct   AS `Predicted default %`,
       actual_default_pct AS `Actual default %`
FROM workspace.lending_club.gold_calibration
ORDER BY decile;


-- ds_risk_bands  (table widget, or bar chart of actual_default_pct by risk_band)
SELECT risk_band,
       loans,
       avg_predicted_pd_pct,
       actual_default_pct,
       avg_interest_rate,
       net_return_pct
FROM workspace.lending_club.gold_risk_bands
ORDER BY risk_band;


-- ds_grades  (bar chart: X grade, Y actual_default_pct)  - the bank's own view, for comparison
SELECT grade,
       COUNT(*)                         AS loans,
       ROUND(100 * AVG(is_default), 1)  AS actual_default_pct,
       ROUND(AVG(int_rate), 2)          AS avg_interest_rate
FROM workspace.lending_club.gold_predictions
WHERE split = 'test'
GROUP BY grade
ORDER BY grade;


-- ds_feature_importance  (horizontal bar: Y feature, X both share columns)
SELECT feature,
       share_pct      AS `SHAP share %`,
       gain_share_pct AS `Gain share %`
FROM workspace.lending_club.gold_shap_importance
ORDER BY mean_abs_shap DESC
LIMIT 15;


-- ds_expected_loss  (2 counters + small table)
SELECT lgd_pct,
       funded_total,
       expected_loss,
       actual_loss,
       expected_loss_pct_of_funded,
       actual_loss_pct_of_funded
FROM workspace.lending_club.gold_expected_loss;


-- ds_reason_codes  (table widget)
SELECT example, loan_id, predicted_pd_pct, rank, feature, feature_value, shap_log_odds
FROM workspace.lending_club.gold_reason_codes
ORDER BY predicted_pd_pct, rank;