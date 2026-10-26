CREATE OR REPLACE VIEW customers_features AS
SELECT
    customer_id, gender, seniorcitizen, partner, dependents, tenure,
    phoneservice, multiplelines, internetservice,
    onlinesecurity, onlinebackup, deviceprotection, techsupport,
    streamingtv, streamingmovies, contract, paperlessbilling, paymentmethod,
    monthlycharges, totalcharges, churn,
    ( CASE WHEN onlinesecurity   = 'Yes' THEN 1 ELSE 0 END
    + CASE WHEN onlinebackup     = 'Yes' THEN 1 ELSE 0 END
    + CASE WHEN deviceprotection = 'Yes' THEN 1 ELSE 0 END
    + CASE WHEN techsupport      = 'Yes' THEN 1 ELSE 0 END
    + CASE WHEN streamingtv      = 'Yes' THEN 1 ELSE 0 END
    + CASE WHEN streamingmovies  = 'Yes' THEN 1 ELSE 0 END ) AS num_addons,
    CASE WHEN internetservice <> 'No' AND techsupport <> 'Yes'
         THEN 1 ELSE 0 END AS no_tech_support,
    CASE WHEN tenure > 0 THEN totalcharges / tenure
         ELSE monthlycharges END AS avg_monthly_spend,
    CASE WHEN tenure > 0 THEN monthlycharges - totalcharges / tenure
         ELSE 0 END AS charge_trend,
    CASE WHEN tenure <= 12 THEN '0-12'
         WHEN tenure <= 24 THEN '13-24'
         WHEN tenure <= 48 THEN '25-48'
         ELSE '49+' END AS tenure_bucket
FROM customers_raw;