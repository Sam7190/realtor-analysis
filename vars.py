# Data
supply_data_path = 'data\RDC_Inventory_Core_Metrics_Zip_History.csv'
demand_data_path = 'data\RDC_Inventory_Hotness_Metrics_Zip_History.csv'
zip2county = 'data/ZIP_COUNTY_032019.xlsx'

# Minimum Records
start_date = '2025-07'
end_date = '2017-07'
check_window_months = 36
min_acceptable_records = 32

# Default Configs
high_risk_price_change = -0.02
high_risk_days_on_market = 90
high_risk_view_count_change = -0.1

strong_market_price_change = 0.02
strong_market_days_on_market = 30
strong_market_view_count_change = 0.1

max_months_predicted = 12

# Features
features = ['median_listing_price',
       'median_listing_price_mm', 'median_listing_price_yy',
       'active_listing_count', 'active_listing_count_mm',
       'active_listing_count_yy', 'median_days_on_market',
       'median_days_on_market_mm', 'median_days_on_market_yy',
       'new_listing_count', 'new_listing_count_mm', 'new_listing_count_yy',
       'price_increased_count', 'price_increased_count_mm',
       'price_increased_count_yy', 'price_increased_share',
       'price_increased_share_mm', 'price_increased_share_yy',
       'price_reduced_count', 'price_reduced_count_mm',
       'price_reduced_count_yy', 'price_reduced_share',
       'price_reduced_share_mm', 'price_reduced_share_yy',
       'pending_listing_count', 'pending_listing_count_mm',
       'pending_listing_count_yy', 'median_listing_price_per_square_foot',
       'median_listing_price_per_square_foot_mm',
       'median_listing_price_per_square_foot_yy', 'median_square_feet',
       'median_square_feet_mm', 'median_square_feet_yy',
       'average_listing_price', 'average_listing_price_mm',
       'average_listing_price_yy', 'total_listing_count',
       'total_listing_count_mm', 'total_listing_count_yy', 'pending_ratio',
       'pending_ratio_mm', 'pending_ratio_yy',
       'hh_rank', 'hotness_rank', 'hotness_rank_mm', 'hotness_rank_yy', 'hotness_score',
       'supply_score', 'demand_score',
       'median_dom_mm_day', 'median_dom_yy_day', 'median_dom_vs_us',
       'page_view_count_per_property_mm', 'page_view_count_per_property_yy',
       'page_view_count_per_property_vs_us', 'median_listing_price_vs_us']

outcomes = ['high_risk_price_change',
            'strong_market_price_change',
            'high_risk_days_on_market',
            'strong_market_days_on_market',
            'high_risk_view_count_change',
            'strong_market_view_count_change']

out_names = ['price_change', 'days_on_market', 'view_count_change']

# Fixed factors
outcome_thresholds = {
    'high_risk': {
        'price_change': high_risk_price_change,
        'days_on_market': high_risk_days_on_market,
        'view_count_change': high_risk_view_count_change
    },
    'strong_market': {
        'price_change': strong_market_price_change,
        'days_on_market': strong_market_days_on_market,
        'view_count_change': strong_market_view_count_change
    }
}