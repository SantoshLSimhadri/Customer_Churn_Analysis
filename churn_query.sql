# Main query to extract customer churn dataset
WITH customer_base AS (
    -- Get all customers with their basic information
    SELECT 
        c.customer_id,
        c.gender,
        c.age,
        c.senior_citizen,
        c.partner,
        c.dependents,
        c.tenure_months,
        c.phone_service,
        c.multiple_lines,
        c.internet_service,
        c.online_security,
        c.online_backup,
        c.device_protection,
        c.tech_support,
        c.streaming_tv,
        c.streaming_movies,
        c.contract_type,
        c.paperless_billing,
        c.payment_method,
        c.monthly_charges,
        c.total_charges,
        c.churn_flag,
        c.created_date,
        c.last_update_date
    FROM customers c
    WHERE c.created_date >= DATEADD(YEAR, -3, GETDATE()) -- Last 3 years of data
),

-- Calculate usage metrics
usage_metrics AS (
    SELECT 
        cb.customer_id,
        COALESCE(um.avg_monthly_minutes, 0) as avg_monthly_minutes,
        COALESCE(um.avg_monthly_data_gb, 0) as avg_monthly_data_gb,
        COALESCE(um.total_service_calls, 0) as total_service_calls,
        COALESCE(um.avg_monthly_sms, 0) as avg_monthly_sms,
        CASE 
            WHEN um.avg_monthly_data_gb > 10 THEN 'Heavy'
            WHEN um.avg_monthly_data_gb > 2 THEN 'Medium'
            ELSE 'Light'
        END as data_usage_category
    FROM customer_base cb
    LEFT JOIN (
        SELECT 
            u.customer_id,
            AVG(u.monthly_minutes) as avg_monthly_minutes,
            AVG(u.monthly_data_usage_gb) as avg_monthly_data_gb,
            AVG(u.monthly_sms_count) as avg_monthly_sms,
            COUNT(DISTINCT u.service_call_id) as total_service_calls
        FROM usage_data u
        WHERE u.usage_date >= DATEADD(MONTH, -12, GETDATE())
        GROUP BY u.customer_id
    ) um ON cb.customer_id = um.customer_id
),

-- Payment and billing history
payment_metrics AS (
    SELECT 
        cb.customer_id,
        COALESCE(pm.late_payments_count, 0) as late_payments_count,
        COALESCE(pm.avg_days_late, 0) as avg_days_late,
        COALESCE(pm.total_payments, 0) as total_payments,
        COALESCE(pm.payment_failures, 0) as payment_failures,
        CASE 
            WHEN pm.late_payments_count > 3 THEN 'High Risk'
            WHEN pm.late_payments_count > 1 THEN 'Medium Risk'
            ELSE 'Low Risk'
        END as payment_risk_category
    FROM customer_base cb
    LEFT JOIN (
        SELECT 
            p.customer_id,
            SUM(CASE WHEN p.payment_status = 'Late' THEN 1 ELSE 0 END) as late_payments_count,
            AVG(CASE WHEN p.payment_status = 'Late' THEN p.days_late ELSE 0 END) as avg_days_late,
            COUNT(p.payment_id) as total_payments,
            SUM(CASE WHEN p.payment_status = 'Failed' THEN 1 ELSE 0 END) as payment_failures
        FROM payments p
        WHERE p.payment_date >= DATEADD(YEAR, -2, GETDATE())
        GROUP BY p.customer_id
    ) pm ON cb.customer_id = pm.customer_id
),

-- Customer service interactions
service_metrics AS (
    SELECT 
        cb.customer_id,
        COALESCE(sm.total_tickets, 0) as total_support_tickets,
        COALESCE(sm.avg_resolution_days, 0) as avg_resolution_days,
        COALESCE(sm.escalated_tickets, 0) as escalated_tickets,
        COALESCE(sm.satisfaction_score, 5) as avg_satisfaction_score,
        CASE 
            WHEN sm.escalated_tickets > 2 THEN 'High Maintenance'
            WHEN sm.total_tickets > 5 THEN 'Medium Maintenance'
            ELSE 'Low Maintenance'
        END as service_usage_category
    FROM customer_base cb
    LEFT JOIN (
        SELECT 
            st.customer_id,
            COUNT(st.ticket_id) as total_tickets,
            AVG(DATEDIFF(DAY, st.created_date, st.resolved_date)) as avg_resolution_days,
            SUM(CASE WHEN st.escalated = 1 THEN 1 ELSE 0 END) as escalated_tickets,
            AVG(st.satisfaction_rating) as satisfaction_score
        FROM support_tickets st
        WHERE st.created_date >= DATEADD(YEAR, -1, GETDATE())
        GROUP BY st.customer_id
    ) sm ON cb.customer_id = sm.customer_id
),

-- Competition and market analysis
market_metrics AS (
    SELECT 
        cb.customer_id,
        COALESCE(mm.competitor_offers_received, 0) as competitor_offers_received,
        COALESCE(mm.price_inquiries, 0) as price_inquiries,
        COALESCE(mm.plan_changes, 0) as plan_changes_last_year,
        CASE 
            WHEN mm.competitor_offers_received > 2 THEN 'High Competition Exposure'
            WHEN mm.competitor_offers_received > 0 THEN 'Medium Competition Exposure'
            ELSE 'Low Competition Exposure'
        END as competition_exposure
    FROM customer_base cb
    LEFT JOIN (
        SELECT 
            ci.customer_id,
            COUNT(CASE WHEN ci.interaction_type = 'Competitor Offer' THEN 1 END) as competitor_offers_received,
            COUNT(CASE WHEN ci.interaction_type = 'Price Inquiry' THEN 1 END) as price_inquiries,
            COUNT(CASE WHEN ci.interaction_type = 'Plan Change' THEN 1 END) as plan_changes
        FROM customer_interactions ci
        WHERE ci.interaction_date >= DATEADD(YEAR, -1, GETDATE())
        GROUP BY ci.customer_id
    ) mm ON cb.customer_id = mm.customer_id
),

-- Recent activity and engagement
engagement_metrics AS (
    SELECT 
        cb.customer_id,
        DATEDIFF(DAY, em.last_login_date, GETDATE()) as days_since_last_login,
        COALESCE(em.monthly_app_sessions, 0) as avg_monthly_app_sessions,
        COALESCE(em.feature_usage_score, 0) as feature_usage_score,
        CASE 
            WHEN DATEDIFF(DAY, em.last_login_date, GETDATE()) > 30 THEN 'Inactive'
            WHEN DATEDIFF(DAY, em.last_login_date, GETDATE()) > 7 THEN 'Low Activity'
            ELSE 'Active'
        END as engagement_level
    FROM customer_base cb
    LEFT JOIN (
        SELECT 
            ea.customer_id,
            MAX(ea.login_date) as last_login_date,
            AVG(ea.monthly_sessions) as monthly_app_sessions,
            AVG(ea.features_used_count) as feature_usage_score
        FROM engagement_analytics ea
        WHERE ea.activity_date >= DATEADD(MONTH, -6, GETDATE())
        GROUP BY ea.customer_id
    ) em ON cb.customer_id = em.customer_id
)

-- Final consolidated dataset for churn analysis
SELECT 
    -- Customer Demographics
    cb.customer_id,
    cb.gender,
    cb.age,
    cb.senior_citizen,
    cb.partner,
    cb.dependents,
    
    -- Service Information
    cb.tenure_months,
    cb.phone_service,
    cb.multiple_lines,
    cb.internet_service,
    cb.online_security,
    cb.online_backup,
    cb.device_protection,
    cb.tech_support,
    cb.streaming_tv,
    cb.streaming_movies,
    
    -- Contract and Billing
    cb.contract_type,
    cb.paperless_billing,
    cb.payment_method,
    cb.monthly_charges,
    cb.total_charges,
    
    -- Usage Metrics
    um.avg_monthly_minutes,
    um.avg_monthly_data_gb,
    um.avg_monthly_sms,
    um.total_service_calls,
    um.data_usage_category,
    
    -- Payment Behavior
    pm.late_payments_count,
    pm.avg_days_late,
    pm.payment_failures,
    pm.payment_risk_category,
    
    -- Customer Service
    sm.total_support_tickets,
    sm.avg_resolution_days,
    sm.escalated_tickets,
    sm.avg_satisfaction_score,
    sm.service_usage_category,
    
    -- Market Competition
    mm.competitor_offers_received,
    mm.price_inquiries,
    mm.plan_changes_last_year,
    mm.competition_exposure,
    
    -- Engagement
    em.days_since_last_login,
    em.avg_monthly_app_sessions,
    em.feature_usage_score,
    em.engagement_level,
    
    -- Calculated Features
    CASE 
        WHEN cb.tenure_months <= 12 THEN 'New Customer'
        WHEN cb.tenure_months <= 36 THEN 'Established Customer'
        ELSE 'Long-term Customer'
    END as customer_lifetime_category,
    
    CASE 
        WHEN cb.monthly_charges < 50 THEN 'Low Value'
        WHEN cb.monthly_charges < 100 THEN 'Medium Value'
        ELSE 'High Value'
    END as customer_value_segment,
    
    cb.monthly_charges * cb.tenure_months as lifetime_value,
    
    -- Target Variable
    cb.churn_flag as churn,
    
    -- Metadata
    cb.created_date,
    cb.last_update_date,
    GETDATE() as extraction_date

FROM customer_base cb
LEFT JOIN usage_metrics um ON cb.customer_id = um.customer_id
LEFT JOIN payment_metrics pm ON cb.customer_id = pm.customer_id
LEFT JOIN service_metrics sm ON cb.customer_id = sm.customer_id
LEFT JOIN market_metrics mm ON cb.customer_id = mm.customer_id
LEFT JOIN engagement_metrics em ON cb.customer_id = em.customer_id

-- Filter for complete records
WHERE cb.monthly_charges IS NOT NULL
  AND cb.total_charges IS NOT NULL
  AND cb.tenure_months >= 1  -- Exclude customers with less than 1 month tenure

ORDER BY cb.customer_id;

-- =====================================================
-- Additional Queries for Data Quality and Exploration
-- =====================================================

-- Data Quality Check
SELECT 
    'Total Customers' as metric,
    COUNT(*) as count,
    '' as percentage
FROM customers
WHERE created_date >= DATEADD(YEAR, -3, GETDATE())

UNION ALL

SELECT 
    'Customers with Churn Flag',
    COUNT(*),
    CAST(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM customers WHERE created_date >= DATEADD(YEAR, -3, GETDATE())) AS VARCHAR(10)) + '%'
FROM customers 
WHERE churn_flag IS NOT NULL 
  AND created_date >= DATEADD(YEAR, -3, GETDATE())

UNION ALL

SELECT 
    'Churned Customers',
    COUNT(*),
    CAST(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM customers WHERE created_date >= DATEADD(YEAR, -3, GETDATE())) AS VARCHAR(10)) + '%'
FROM customers 
WHERE churn_flag = 1 
  AND created_date >= DATEADD(YEAR, -3, GETDATE());

-- Monthly Churn Rate Trend
SELECT 
    YEAR(last_update_date) as year,
    MONTH(last_update_date) as month,
    COUNT(*) as total_customers,
    SUM(churn_flag) as churned_customers,
    CAST(SUM(churn_flag) * 100.0 / COUNT(*) AS DECIMAL(5,2)) as churn_rate_percent
FROM customers
WHERE last_update_date >= DATEADD(YEAR, -2, GETDATE())
GROUP BY YEAR(last_update_date), MONTH(last_update_date)
ORDER BY year, month;

-- Feature Distribution Analysis
SELECT 
    'Contract Type' as feature,
    contract_type as value,
    COUNT(*) as customer_count,
    SUM(churn_flag) as churned_count,
    CAST(SUM(churn_flag) * 100.0 / COUNT(*) AS DECIMAL(5,2)) as churn_rate
FROM customers
WHERE created_date >= DATEADD(YEAR, -3, GETDATE())
GROUP BY contract_type

UNION ALL

SELECT 
    'Payment Method',
    payment_method,
    COUNT(*),
    SUM(churn_flag),
    CAST(SUM(churn_flag) * 100.0 / COUNT(*) AS DECIMAL(5,2))
FROM customers
WHERE created_date >= DATEADD(YEAR, -3, GETDATE())
GROUP BY payment_method

UNION ALL

SELECT 
    'Internet Service',
    internet_service,
    COUNT(*),
    SUM(churn_flag),
    CAST(SUM(churn_flag) * 100.0 / COUNT(*) AS DECIMAL(5,2))
FROM customers
WHERE created_date >= DATEADD(YEAR, -3, GETDATE())
GROUP BY internet_service

ORDER BY feature, churn_rate DESC;
