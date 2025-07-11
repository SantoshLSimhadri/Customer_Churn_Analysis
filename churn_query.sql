-- SQL Example: Extract active customers in the last 30 days
SELECT * FROM customers WHERE last_active >= CURRENT_DATE - INTERVAL '30 days';