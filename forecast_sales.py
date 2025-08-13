import pandas as pd
import numpy as np
from prophet import Prophet
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from jinja2 import Environment, FileSystemLoader, TemplateError
import os
from datetime import datetime
import json
import argparse
import pickle
from sklearn.model_selection import ParameterGrid
import logging
import yaml
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)


from insight_modules import (compute_feature_importance, clean_data_for_json, format_df,
                            generate_alerts, load_holidays, 
                            generate_business_strategies, generate_interesting_fact, 
                            generate_summary_txt, get_default_result)

def load_config():
    try:
        with open('config.yaml', 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        print(f"[v0.0.0] Error loading config.yaml: {e}")
        return {'version': '0.0.0'}

CONFIG = load_config()
VERSION = CONFIG['version']
INPUT_FILE = CONFIG['input_file']
OUTPUT_DIR = CONFIG['output_dir']
FEATURES = CONFIG['features']['tree_features']
PROPHET_REGRESSORS = CONFIG['features']['prophet_regressors']
REQUIRED_COLUMNS = CONFIG['data']['required_columns']
COLUMNS_TO_DROP = CONFIG['data']['columns_to_drop']
HOLIDAY_COLUMNS = CONFIG['data']['holiday_columns']

ZERO_SALE_THRESHOLDS = CONFIG.get('features', {}).get('zero_sale_thresholds', {'campaign_intensity': 0.3, 'lag_1': 0.1})
HOLDOUT_FEATURE_STRATEGY = CONFIG.get('features', {}).get('holdout_feature_strategy', 'static')

def set_zero_sale_indicator(holiday, campaign_intensity, lag_1):
    return int((holiday == 0) and
               (campaign_intensity < ZERO_SALE_THRESHOLDS['campaign_intensity']) and
               (lag_1 < ZERO_SALE_THRESHOLDS['lag_1']))

def load_data():
    print(f"[v{VERSION}] Loading data from {INPUT_FILE}")
    try:
        df = pd.read_csv(INPUT_FILE)
        print(f"[v{VERSION}] Raw data: {len(df)} rows, columns: {list(df.columns)}")
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
        df['product_id'] = df['product_id'].str.upper()
        missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing_columns:
            print(f"[v{VERSION}] Warning: Missing columns {missing_columns}, adding defaults")
            for col in missing_columns:
                if col == 'campaign_intensity':
                    df[col] = 0.0
                elif col == 'season':
                    df[col] = df['date'].dt.month % 12 // 3 + 1
                else:
                    raise ValueError(f"Missing required column: {col}")
        columns_to_drop = [col for col in df.columns if col in COLUMNS_TO_DROP]
        if columns_to_drop:
            df = df.drop(columns=columns_to_drop)
        df['is_zero_sale'] = (df['quantity'] == 0).astype(int)
        print(f"[v{VERSION}] Loaded data: {len(df)} rows, zero-sale ratio: {df['is_zero_sale'].mean():.2f}")
        return df
    except Exception as e:
        print(f"[v{VERSION}] Error loading data: {type(e).__name__}: {str(e)}")
        raise

def prepare_data(df, product, hold_out_start):
    print(f"[v{VERSION}] Preparing data for product: {product}")
    product_df = df[df['product_id'] == product].copy()
    if product_df.empty:
        print(f"[v{VERSION}] No data for product {product}")
        return pd.DataFrame(), None, None
    product_df['date'] = pd.to_datetime(product_df['date'], errors='coerce')
    product_df = product_df.dropna(subset=['date']).sort_values('date')
    holidays = load_holidays()
    holidays = holidays[['ds', 'holiday']].rename(columns={'ds': 'date'})
    holidays['date'] = pd.to_datetime(holidays['date'], errors='coerce')
    holidays = holidays.dropna(subset=['date'])
    holidays['holiday'] = holidays['holiday'].notna().astype(int)
    product_df = product_df.merge(holidays[['date', 'holiday']], on='date', how='left')
    product_df['holiday'] = product_df['holiday'].fillna(0).astype(int)
    product_df['year'] = product_df['date'].dt.year
    product_df['month'] = product_df['date'].dt.month
    product_df['day'] = product_df['date'].dt.day
    train_df = product_df[product_df['date'] < hold_out_start].copy()
    holdout_df = product_df[product_df['date'] >= hold_out_start].copy()
    train_df['lag_1'] = train_df['quantity'].shift(1).fillna(0)
    train_df['lag_7'] = train_df['quantity'].shift(7).fillna(0)
    train_df['rolling_mean_7'] = train_df['quantity'].rolling(window=7).mean().fillna(train_df['quantity'].mean())
    train_df['rolling_std_7'] = train_df['quantity'].rolling(window=7).std().fillna(train_df['quantity'].std())
    train_df['lag_1_missing'] = train_df['lag_1'].isna().astype(int)
    train_df['lag_7_missing'] = train_df['lag_7'].isna().astype(int)
    train_df['zero_sale_indicator'] = train_df.apply(
        lambda row: set_zero_sale_indicator(row['holiday'], row['campaign_intensity'], row['lag_1']), axis=1
    )
    if HOLDOUT_FEATURE_STRATEGY == 'dynamic':
        holdout_df['lag_1'] = 0.0
        holdout_df['lag_7'] = 0.0
        holdout_df['rolling_mean_7'] = train_df['quantity'].mean() if not train_df.empty else 0.0
        holdout_df['rolling_std_7'] = train_df['quantity'].std() if not train_df.empty else 0.0
        holdout_df['lag_1_missing'] = 1
        holdout_df['lag_7_missing'] = 1
        holdout_df['zero_sale_indicator'] = holdout_df.apply(
            lambda row: set_zero_sale_indicator(row['holiday'], row['campaign_intensity'], row['lag_1']), axis=1
        )
    else:
        holdout_df['lag_1'] = train_df['lag_1'].mean() if not train_df.empty else 0.0
        holdout_df['lag_7'] = train_df['lag_7'].mean() if not train_df.empty else 0.0
        holdout_df['rolling_mean_7'] = train_df['rolling_mean_7'].mean() if not train_df.empty else 0.0
        holdout_df['rolling_std_7'] = train_df['rolling_std_7'].mean() if not train_df.empty else 0.0
        holdout_df['lag_1_missing'] = train_df['lag_1_missing'].mean() if not train_df.empty else 1
        holdout_df['lag_7_missing'] = train_df['lag_7_missing'].mean() if not train_df.empty else 1
        holdout_df['zero_sale_indicator'] = train_df['zero_sale_indicator'].mean().round() if not train_df.empty else 0
    product_df = pd.concat([train_df, holdout_df]).sort_values('date')
    return product_df, None, None

def estimate_future_regressors(df, forecast_start, forecast_end):
    if df.empty:
        print(f"[v{VERSION}] No data for future regressors, using defaults")
        future_dates = pd.date_range(start=forecast_start, end=forecast_end, freq='D')
        return pd.DataFrame({
            'ds': future_dates,
            'campaign_intensity': [0.0] * len(future_dates),
            'holiday': [0] * len(future_dates),
            'season': [0] * len(future_dates),
            'zero_sale_indicator': [0] * len(future_dates)
        })
    df_last_year = df[df['date'] >= df['date'].max() - pd.Timedelta(days=365)].copy()
    if df_last_year.empty:
        print(f"[v{VERSION}] No historical data for future regressors, using defaults")
        future_dates = pd.date_range(start=forecast_start, end=forecast_end, freq='D')
        return pd.DataFrame({
            'ds': future_dates,
            'campaign_intensity': [0.0] * len(future_dates),
            'holiday': [0] * len(future_dates),
            'season': [0] * len(future_dates),
            'zero_sale_indicator': [0] * len(future_dates)
        })
    df_last_year['month_day'] = df_last_year['date'].dt.strftime('%m-%d')
    campaign_avg = df_last_year.groupby('month_day')['campaign_intensity'].mean().reset_index()
    season_map = df_last_year.groupby('month_day')['season'].first().reset_index()
    # Calculate zero_sale_indicator for historical data
    df_last_year['zero_sale_indicator'] = df_last_year.apply(
        lambda row: set_zero_sale_indicator(row.get('holiday', 0), row['campaign_intensity'], row['quantity']), axis=1
    )
    zero_sale_avg = df_last_year.groupby('month_day')['zero_sale_indicator'].mean().reset_index()
    holidays = load_holidays()
    holidays = holidays[['ds', 'holiday']]
    holidays['ds'] = pd.to_datetime(holidays['ds'], errors='coerce')
    holidays = holidays.dropna(subset=['ds'])
    holidays = holidays[holidays['ds'].between(forecast_start, forecast_end)]
    holidays['month_day'] = holidays['ds'].dt.strftime('%m-%d')
    holidays['holiday'] = holidays['holiday'].notna().astype(int)
    holiday_avg = holidays.groupby('month_day')['holiday'].max().reset_index()
    future_dates = pd.date_range(start=forecast_start, end=forecast_end, freq='D')
    future_df = pd.DataFrame({'ds': future_dates})
    future_df['month_day'] = future_df['ds'].dt.strftime('%m-%d')
    future_df = future_df.merge(campaign_avg, on='month_day', how='left')
    future_df = future_df.merge(holiday_avg, on='month_day', how='left')
    future_df = future_df.merge(season_map, on='month_day', how='left')
    future_df = future_df.merge(zero_sale_avg, on='month_day', how='left')
    # Use overall historical averages if merge fails
    future_df['campaign_intensity'] = future_df['campaign_intensity'].fillna(df['campaign_intensity'].mean() if not df['campaign_intensity'].isna().all() else 0.0)
    future_df['holiday'] = future_df['holiday'].fillna(0).astype(int)
    future_df['season'] = future_df['season'].fillna(df['season'].mode()[0] if not df['season'].isna().all() and not df['season'].mode().empty else 0)
    future_df['zero_sale_indicator'] = future_df['zero_sale_indicator'].fillna(df_last_year['zero_sale_indicator'].mean() if not df_last_year.empty else 0)
    if future_df[['campaign_intensity', 'holiday', 'season', 'zero_sale_indicator']].isna().all().any():
        print(f"[v{VERSION}] Some regressors missing for forecast period, using historical averages")
    return future_df[['ds', 'campaign_intensity', 'holiday', 'season', 'zero_sale_indicator']]

def prepare_future_features(future_df, i, last_quantity, recent_quantities, predictions=None):
    if i == 0:
        future_df.loc[i, 'lag_1'] = last_quantity
        future_df.loc[i, 'lag_7'] = recent_quantities[-7] if len(recent_quantities) >= 7 else 0.0
        future_df.loc[i, 'rolling_mean_7'] = np.mean(recent_quantities) if recent_quantities else 0.0
        future_df.loc[i, 'rolling_std_7'] = np.std(recent_quantities) if recent_quantities else 0.0
        future_df.loc[i, 'lag_1_missing'] = 6
        future_df.loc[i, 'lag_7_missing'] = 0 if len(recent_quantities) >= 7 else 1
        future_df.loc[i, 'zero_sale_indicator'] = set_zero_sale_indicator(
            future_df.loc[i, 'holiday'], future_df.loc[i, 'campaign_intensity'], future_df.loc[i, 'lag_1']
        )
    else:
        future_df.loc[i, 'lag_1'] = predictions[i-1] if predictions is not None and i-1 < len(predictions) else last_quantity
        future_df.loc[i, 'lag_7'] = predictions[i-7] if predictions is not None and i >= 7 else recent_quantities[i-7] if i < len(recent_quantities) else 0.0
        recent_quantities = (recent_quantities[1:] + [predictions[i-1] if predictions is not None else last_quantity])[-7:]
        future_df.loc[i, 'rolling_mean_7'] = np.mean(recent_quantities)
        future_df.loc[i, 'rolling_std_7'] = np.std(recent_quantities)
        future_df.loc[i, 'lag_1_missing'] = 0
        future_df.loc[i, 'lag_7_missing'] = 0 if i >= 7 or i < len(recent_quantities) else 1
        future_df.loc[i, 'zero_sale_indicator'] = set_zero_sale_indicator(
            future_df.loc[i, 'holiday'], future_df.loc[i, 'campaign_intensity'], future_df.loc[i, 'lag_1']
        )
    return recent_quantities

def tune_model(model_type, product_df, hold_out_start, hold_out_end, forecast_start, forecast_end, config, force_retrain=False):
    product = product_df['product_id'].iloc[0]
    print(f"[v{VERSION}] Tuning {model_type} for product {product}")
    cached_params = load_tuned_params(product, model_type)
    if cached_params and not force_retrain:
        print(f"[v{VERSION}] Loaded cached {model_type} parameters for {product}")
        return cached_params
    param_grid = config['model_params'].get(model_type.lower(), {})
    if not param_grid:
        print(f"[v{VERSION}] No tuning parameters for {model_type} in config, using defaults")
        return {}
    best_params = None
    best_smape = float('inf')
    train_df = product_df[product_df['date'] < hold_out_start]
    val_df = product_df[product_df['date'].between(hold_out_start, hold_out_end)]
    if train_df.empty or val_df.empty:
        print(f"[v{VERSION}] Insufficient data for tuning {model_type} for {product}: train={len(train_df)}, val={len(val_df)}")
        return {}
    if model_type == 'Prophet':
        for params in ParameterGrid(param_grid):
            model = Prophet(
                changepoint_prior_scale=params.get('changepoint_prior_scale', 0.05),
                seasonality_prior_scale=params.get('seasonality_prior_scale', 10.0),
                holidays_prior_scale=params.get('holidays_prior_scale', 10.0),
                daily_seasonality=params.get('daily_seasonality', True),
                weekly_seasonality=params.get('weekly_seasonality', True),
                yearly_seasonality=params.get('yearly_seasonality', True)
            )
            holidays = load_holidays()
            model.add_country_holidays(country_name='BR')
            for feature in ['campaign_intensity', 'season', 'zero_sale_indicator']:
                model.add_regressor(feature)
            model.fit(train_df.rename(columns={'date': 'ds', 'quantity': 'y'}))
            future_dates = pd.date_range(start=hold_out_start, end=hold_out_end, freq='D')
            future_df = pd.DataFrame({'ds': future_dates})
            future_df = future_df.merge(product_df[['date', 'campaign_intensity', 'season', 'zero_sale_indicator']].rename(columns={'date': 'ds'}), on='ds', how='left')
            future_df.fillna(0, inplace=True)
            forecast = model.predict(future_df)
            merged = pd.merge(forecast, val_df[['date', 'quantity']].rename(columns={'date': 'ds', 'quantity': 'y'}), on='ds', how='left')
            y_true = merged['y'].dropna()
            y_pred = merged['yhat'][merged['y'].notnull()]
            if len(y_true) >= 2:
                smape_value = smape(y_true, y_pred)
                if smape_value < best_smape:
                    best_smape = smape_value
                    best_params = params
    else:
        for params in ParameterGrid(param_grid):
            if model_type == 'XGBoost':
                model = XGBRegressor(**params, random_state=42)
            elif model_type == 'LightGBM':
                model = LGBMRegressor(**params, random_state=42)
            else:
                model = CatBoostRegressor(**params, random_state=42, verbose=False)
            model.fit(train_df[FEATURES], train_df['quantity'])
            future_dates = pd.date_range(start=hold_out_start, end=hold_out_end, freq='D')
            future_df = pd.DataFrame({'ds': future_dates})
            future_df = future_df.merge(product_df[['date'] + FEATURES].rename(columns={'date': 'ds'}), on='ds', how='left')
            future_df.fillna(0, inplace=True)
            last_quantity = train_df['quantity'].iloc[-1] if not train_df.empty else 0.0
            recent_quantities = list(train_df['quantity'].tail(7)) if len(train_df) >= 7 else [0.0] * 7
            predictions = []
            for i in range(len(future_df)):
                recent_quantities = prepare_future_features(future_df, i, last_quantity, recent_quantities, predictions)
                pred = model.predict(future_df[FEATURES].iloc[[i]])[0]
                pred = np.maximum(pred, 0)
                predictions.append(pred)
            smape_value = smape(val_df['quantity'], predictions[:len(val_df)])
            if smape_value < best_smape:
                best_smape = smape_value
                best_params = params
    if best_params is None:
        print(f"[v{VERSION}] No valid parameters found for {model_type} for {product}, using defaults")
        best_params = {}
    save_tuned_params(product, model_type, best_params)
    print(f"[v{VERSION}] Best parameters for {model_type} for {product}: {best_params}, SMAPE: {best_smape}")
    return best_params

def save_tuned_params(product, model_type, params):
    file_path = os.path.join(OUTPUT_DIR, f'tuned_params_{product}_{model_type}.json')
    with open(file_path, 'w') as f:
        json.dump(params, f)
    print(f"[v{VERSION}] Saved tuned parameters for {model_type} for {product} to {file_path}")

def load_tuned_params(product, model_type):
    file_path = os.path.join(OUTPUT_DIR, f'tuned_params_{product}_{model_type}.json')
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    return None
    
def train_prophet(product_df, hold_out_start, hold_out_end, forecast_start, forecast_end, config, force_retrain=False):
    product_id = product_df['product_id'].iloc[0] if not product_df.empty else 'unknown'
    print(f"[v{VERSION}] Training Prophet model for {product_id}")
    train_df = product_df[product_df['date'] < hold_out_start][['date', 'quantity', 'campaign_intensity', 'season', 'zero_sale_indicator']].copy()
    train_df = train_df.rename(columns={'date': 'ds', 'quantity': 'y'})
    future_dates = pd.date_range(start=hold_out_start, end=forecast_end, freq='D')
    if len(train_df) < 2:
        print(f"[v{VERSION}] Insufficient data for Prophet for product {product_id}: {len(train_df)} rows")
        mean_quantity = product_df[product_df['date'].between(hold_out_start, hold_out_end)]['quantity'].mean() if not product_df.empty else 0.0
        return None, pd.DataFrame({
            'ds': future_dates,
            'yhat': [mean_quantity] * len(future_dates),
            'yhat_lower': [mean_quantity * 0.8] * len(future_dates),
            'yhat_upper': [mean_quantity * 1.2] * len(future_dates)
        })
    best_params = tune_model('Prophet', product_df, hold_out_start, hold_out_end, forecast_start, forecast_end, config, force_retrain) or {}
    model = Prophet(
        changepoint_prior_scale=best_params.get('changepoint_prior_scale', 0.05),
        seasonality_prior_scale=best_params.get('seasonality_prior_scale', 10.0),
        holidays_prior_scale=best_params.get('holidays_prior_scale', 10.0),
        daily_seasonality=best_params.get('daily_seasonality', True),
        weekly_seasonality=best_params.get('weekly_seasonality', True),
        yearly_seasonality=best_params.get('yearly_seasonality', True)
    )
    holidays = load_holidays()
    model.add_country_holidays(country_name='BR')
    for feature in ['campaign_intensity', 'season', 'zero_sale_indicator']:
        model.add_regressor(feature)
    model.fit(train_df)
    future_df = pd.DataFrame({'ds': future_dates})
    future_df = future_df.merge(product_df[['date', 'campaign_intensity', 'season', 'zero_sale_indicator']].rename(columns={'date': 'ds'}), on='ds', how='left')
    future_regressors = estimate_future_regressors(product_df, forecast_start, forecast_end)
    for i, row in future_df.iterrows():
        if row['ds'] >= forecast_start:
            idx = i - len(pd.date_range(hold_out_start, hold_out_end))
            if idx < len(future_regressors):
                for col in ['campaign_intensity', 'holiday', 'season', 'zero_sale_indicator']:
                    future_df.loc[i, col] = future_regressors[col].iloc[idx]
    future_df.fillna(0, inplace=True)
    forecast = model.predict(future_df)
    forecast['yhat'] = np.maximum(forecast['yhat'], 0)
    print(f"[v{VERSION}] Prophet forecast generated for {product_id}: {len(forecast)} rows")
    return model, forecast[['ds', 'yhat']]


def train_tree(product_df, hold_out_start, hold_out_end, forecast_start, forecast_end, model_type, config, force_retrain=False):
    product_id = product_df['product_id'].iloc[0] if not product_df.empty else 'unknown'
    print(f"[v{VERSION}] Training {model_type.capitalize()} model for {product_id}")
    train_df = product_df[product_df['date'] < hold_out_start].copy()
    future_dates = pd.date_range(start=hold_out_start, end=forecast_end, freq='D')
    if train_df.empty:
        print(f"[v{VERSION}] No training data for {model_type.capitalize()} for product {product_id}")
        return None, pd.DataFrame({
            'ds': future_dates,
            'yhat': [0.0] * len(future_dates)
        })
    best_params = tune_model(model_type.capitalize(), product_df, hold_out_start, hold_out_end, forecast_start, forecast_end, config, force_retrain) or {}
    if model_type == 'xgboost':
        model = XGBRegressor(**best_params, random_state=42)
    elif model_type == 'lightgbm':
        model = LGBMRegressor(**best_params, random_state=42)
    else:
        model = CatBoostRegressor(**best_params, random_state=42, verbose=False)
    model.fit(train_df[FEATURES], train_df['quantity'])
    future_df = pd.DataFrame({'ds': future_dates})
    future_df = future_df.merge(product_df[['date'] + FEATURES].rename(columns={'date': 'ds'}), on='ds', how='left')
    future_regressors = estimate_future_regressors(product_df, forecast_start, forecast_end)
    for i, row in future_df.iterrows():
        if row['ds'] >= forecast_start:
            idx = i - len(pd.date_range(hold_out_start, hold_out_end))
            if idx < len(future_regressors):
                for col in ['campaign_intensity', 'holiday', 'season', 'zero_sale_indicator']:
                    future_df.loc[i, col] = future_regressors[col].iloc[idx]
    future_df.fillna(0, inplace=True)
    last_quantity = train_df['quantity'].iloc[-1] if not train_df.empty else 0.0
    recent_quantities = list(train_df['quantity'].tail(7)) if len(train_df) >= 7 else [0.0] * 7
    predictions = []
    for i in range(len(future_df)):
        recent_quantities = prepare_future_features(future_df, i, last_quantity, recent_quantities, predictions)
        pred = model.predict(future_df[FEATURES].iloc[[i]])[0]
        pred = np.maximum(pred, 0)
        predictions.append(pred)
    forecast = pd.DataFrame({
        'ds': future_dates,
        'yhat': predictions
    })
    print(f"[v{VERSION}] {model_type.capitalize()} forecast generated for {product_id}: {len(forecast)} rows")
    return model, forecast

def save_daily_predictions(product, model_name, predictions, product_df, hold_out_start, hold_out_end, forecast_start, forecast_end, sparse_products=None):
    if sparse_products is None:
        sparse_products = []
    daily_data = []
    expected_rows = len(pd.date_range(start=hold_out_start, end=forecast_end, freq='D'))
    if predictions.empty or 'ds' not in predictions.columns or 'yhat' not in predictions.columns or len(predictions) != expected_rows:
        print(f"[v{VERSION}] Warning: Invalid predictions for {model_name} ({product}): {len(predictions)} rows, expected {expected_rows}")
        future_dates = pd.date_range(start=hold_out_start, end=forecast_end, freq='D')
        predictions = pd.DataFrame({
            'ds': future_dates,
            'yhat': [0.0] * len(future_dates)
        })
    predictions = predictions.copy()
    predictions['ds'] = pd.to_datetime(predictions['ds'])
    predictions['Product'] = product
    predictions['Date'] = predictions['ds'].dt.strftime('%Y-%m-%d')
    predictions['Model'] = model_name
    predictions['Period'] = predictions['ds'].apply(
        lambda x: 'Holdout' if hold_out_start <= x <= hold_out_end else 'Forecast'
    )
    holdout_regressors = product_df[product_df['date'].between(hold_out_start, hold_out_end)][['date', 'zero_sale_indicator']].rename(columns={'date': 'ds'})
    if holdout_regressors.empty or 'zero_sale_indicator' not in product_df.columns:
        print(f"[v{VERSION}] No holdout regressors for {product}, calculating from data")
        holdout_df = product_df[product_df['date'].between(hold_out_start, hold_out_end)].copy()
        holdout_df['zero_sale_indicator'] = holdout_df.apply(
            lambda row: set_zero_sale_indicator(row.get('holiday', 0), row['campaign_intensity'], row['quantity']), axis=1
        )
        holdout_regressors = holdout_df[['date', 'zero_sale_indicator']].rename(columns={'date': 'ds'})
        if holdout_regressors.empty:
            holdout_regressors = pd.DataFrame({'ds': pd.date_range(start=hold_out_start, end=hold_out_end, freq='D'), 'zero_sale_indicator': 0})
    holdout_regressors = holdout_regressors.drop_duplicates(subset=['ds'], keep='first')
    future_regressors = estimate_future_regressors(product_df, forecast_start, forecast_end)
    if future_regressors.empty or 'zero_sale_indicator' not in future_regressors.columns:
        print(f"[v{VERSION}] No future regressors for {product}, using default")
        future_regressors = pd.DataFrame({
            'ds': pd.date_range(start=forecast_start, end=forecast_end, freq='D'),
            'zero_sale_indicator': [0] * len(pd.date_range(start=forecast_start, end=forecast_end, freq='D'))
        })
    future_regressors = future_regressors.drop_duplicates(subset=['ds'], keep='first')
    predictions = predictions.merge(holdout_regressors, on='ds', how='left', suffixes=('', '_holdout'))
    predictions = predictions.merge(future_regressors[['ds', 'zero_sale_indicator']], on='ds', how='left', suffixes=('', '_forecast'))
    predictions['zero_sale_indicator'] = predictions['zero_sale_indicator'].fillna(
        predictions['zero_sale_indicator_forecast'].fillna(0)
    )
    predictions = predictions.drop(columns=['zero_sale_indicator_holdout', 'zero_sale_indicator_forecast'], errors='ignore')
    predictions['yhat'] = predictions.apply(
        lambda row: 0 if row['zero_sale_indicator'] == 1 and row['Period'] == 'Forecast' else row['yhat'],
        axis=1
    )
    predictions['yhat'] = np.where(predictions['yhat'] < 0.1, 0, predictions['yhat'])
    holdout_actuals = product_df[product_df['date'].between(hold_out_start, hold_out_end)][['date', 'quantity']]
    if holdout_actuals.empty or 'quantity' not in product_df.columns:
        print(f"[v{VERSION}] No holdout actuals for {product}, creating default")
        holdout_actuals = pd.DataFrame({
            'ds': pd.date_range(start=hold_out_start, end=hold_out_end, freq='D'),
            'actual': [0.0] * len(pd.date_range(start=hold_out_start, end=hold_out_end))
        })
    else:
        holdout_actuals = holdout_actuals.rename(columns={'date': 'ds', 'quantity': 'actual'})
        holdout_actuals = holdout_actuals.drop_duplicates(subset=['ds'], keep='first')
        holdout_actuals['ds'] = pd.to_datetime(holdout_actuals['ds'])
    predictions = predictions.merge(holdout_actuals[['ds', 'actual']], on='ds', how='left')
    seen = set()
    for _, row in predictions.iterrows():
        key = (row['Product'], row['Date'], row['Model'], row['Period'])
        if key not in seen and row['Model'] != 'Actual':
            daily_data.append({
                'product': row['Product'],
                'date': row['Date'],
                'model': row['Model'],
                'period': row['Period'],
                'actual': float(row['actual']) if pd.notnull(row['actual']) else None,
                'predicted': float(row['yhat'])
            })
            seen.add(key)
    # Skip 'Actual' data for sparse products, handle in main
    if product not in sparse_products and not holdout_actuals.empty:
        for _, row in holdout_actuals.iterrows():
            key = (product, row['ds'].strftime('%Y-%m-%d'), 'Actual', 'Holdout')
            if key not in seen:
                daily_data.append({
                    'product': product,
                    'date': row['ds'].strftime('%Y-%m-%d'),
                    'model': 'Actual',
                    'period': 'Holdout',
                    'actual': float(row['actual']),
                    'predicted': None
                })
                seen.add(key)
    return daily_data

def save_weekly_predictions(product, models, hold_out_start, hold_out_end, forecast_start, forecast_end):
    weekly_data = []
    for model_name, predictions in models.items():
        if predictions.empty or 'ds' not in predictions.columns or 'yhat' not in predictions.columns:
            print(f"[v{VERSION}] Warning: Empty or invalid predictions for {model_name} for product {product} in weekly predictions")
            continue
        period_data = predictions[predictions['ds'].between(hold_out_start, hold_out_end)].copy()
        if not period_data.empty:
            period_data['week'] = period_data['ds'].dt.isocalendar().week
            weekly_avg = period_data.groupby('week')['yhat'].mean().reset_index()
            for _, row in weekly_avg.iterrows():
                weekly_data.append({
                    'Product': product,
                    'Week': int(row['week']),
                    'Model': model_name,
                    'Period': 'Holdout',
                    'Avg_Quantity_Pred': float(row['yhat'])
                })
        forecast_data = predictions[predictions['ds'].between(forecast_start, forecast_end)].copy()
        if not forecast_data.empty:
            forecast_data['week'] = forecast_data['ds'].dt.isocalendar().week
            weekly_avg = forecast_data.groupby('week')['yhat'].mean().reset_index()
            for _, row in weekly_avg.iterrows():
                weekly_data.append({
                    'Product': product,
                    'Week': int(row['week']),
                    'Model': model_name,
                    'Period': 'Forecast',
                    'Avg_Quantity_Pred': float(row['yhat'])
                })
    return weekly_data

def smape(actual, predicted):
    return 100 * np.mean(2 * np.abs(predicted - actual) / (np.abs(actual) + np.abs(predicted) + 1e-10))

def evaluate_model(predictions_df, df_actuals, product_id, model_name, hold_out_start, hold_out_end, forecast_start, forecast_end):
    print(f"[v{VERSION}] Evaluating {model_name} for product {product_id}")
    if 'ds' not in predictions_df.columns or 'yhat' not in predictions_df.columns:
        print(f"[v{VERSION}] Missing required columns in predictions_df for {product_id}: {predictions_df.columns}")
        return None
    print(f"[v{VERSION}] Product {product_id}: Actuals rows={len(df_actuals)}, Predictions rows={len(predictions_df)}")
    if not df_actuals.empty:
        print(f"[v{VERSION}] Product {product_id}: Actuals sample (1 row)={df_actuals.head(1).to_dict()}")
        print(f"[v{VERSION}] Product {product_id}: Predictions sample (1 row)={predictions_df.head(1).to_dict()}")
    df_actuals_product = df_actuals[df_actuals['product_id'] == product_id][['ds', 'y']].copy()
    if df_actuals_product.empty:
        print(f"[v{VERSION}] No actuals data for product {product_id} in holdout period")
        return None
    merged = pd.merge(
        predictions_df,
        df_actuals_product,
        how='left',
        on='ds',
        validate='one_to_one'
    )
    merged.rename(columns={'y': 'actual'}, inplace=True)
    print(f"[v{VERSION}] Product {product_id}: Merged rows={len(merged)}, Actuals non-null={merged['actual'].notnull().sum()}")
    if 'actual' not in merged.columns or merged['actual'].isna().all():
        print(f"[v{VERSION}] No 'actual' column or all actuals are null for {product_id} after merge. Actuals head: {df_actuals_product.head()}")
        return None
    evaluation_data = merged[merged['actual'].notnull()].copy()
    if evaluation_data.empty:
        print(f"[v{VERSION}] No non-null actuals data for {product_id} in evaluation period")
        return None
    holdout_predictions = predictions_df[predictions_df['ds'].between(hold_out_start, hold_out_end)]
    forecast_predictions = predictions_df[predictions_df['ds'] >= forecast_start]
    print(f"[v{VERSION}] Product {product_id}: Holdout prediction dates={holdout_predictions['ds'].min()} to {holdout_predictions['ds'].max()}")
    print(f"[v{VERSION}] Product {product_id}: Forecast prediction dates={forecast_predictions['ds'].min()} to {forecast_predictions['ds'].max()}")
    print(f"[v{VERSION}] Product {product_id}: Actuals dates={df_actuals_product['ds'].min()} to {df_actuals_product['ds'].max()}")
    y_true = evaluation_data['actual']
    y_pred = evaluation_data['yhat']
    if len(y_true) < 2 or np.var(y_true) == 0 or np.var(y_pred) == 0:
        print(f"[v{VERSION}] Insufficient variance or data points for {product_id} ({model_name})")
        return {
            'RMSE': 0.0,
            'R2': 0.0,
            'MAE': mean_absolute_error(y_true, y_pred) if len(y_true) >= 1 else 0.0,
            'SMAPE': smape(y_true, y_pred) if len(y_true) >= 1 else 0.0
        }
    smape_value = smape(y_true, y_pred)
    metrics = {
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R2': r2_score(y_true, y_pred),
        'MAE': mean_absolute_error(y_true, y_pred),
        'SMAPE': smape_value
    }
    print(f"[v{VERSION}] Metrics for {product_id} ({model_name}): {metrics}")
    return metrics

def generate_html_report(timestamp, template_vars):
    template_name = 'template.html'
    possible_dirs = [
        os.path.abspath(os.path.dirname(__file__)),
        os.path.abspath(os.path.join(os.path.dirname(__file__), 'templates')),
        os.path.abspath('.'),
        os.path.abspath('templates')
    ]
    template_path = None
    for dir_path in possible_dirs:
        candidate_path = os.path.join(dir_path, template_name)
        if os.path.exists(candidate_path):
            template_path = candidate_path
            break
    if not template_path:
        error_message = f"Template file '{template_name}' not found in paths: {', '.join(possible_dirs)}"
        print(f"[v{VERSION}] Error loading template: FileNotFoundError: {error_message}")
        html_path = os.path.join(OUTPUT_DIR, f'sales_forecast_report_{timestamp}.html')
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sales Forecast Report</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-100">
    <div class="container mx-auto p-4">
        <div class="bg-white shadow-md rounded-lg p-6">
            <h1 class="text-3xl font-bold mb-4">Sales Forecast Report - {datetime.now().strftime('%B %d, %Y')}</h1>
            <p class="text-gray-600 mb-4">Version: {VERSION}</p>
            <p class="text-red-600 font-bold mb-4">Error: Failed to load template: {error_message}</p>
            <p class="text-red-600 mb-4">Please ensure template.html exists in one of the following paths: {', '.join(possible_dirs)}</p>
        </div>
    </div>
</body>
</html>
""")
        print(f"[v{VERSION}] Generated fallback HTML report: {html_path}")
        return
    try:
        env = Environment(loader=FileSystemLoader(os.path.dirname(template_path)), autoescape=False)
        env.filters['tojson'] = lambda v: json.dumps(v, ensure_ascii=False, allow_nan=True).replace("</", "<\\/")
        env.filters['from_json'] = lambda v: json.loads(v) if isinstance(v, str) else v
        template = env.get_template(os.path.basename(template_path))
        html_content = template.render(**template_vars)
        if '{{' in html_content or '}}' in html_content:
            print(f"[v{VERSION}] Warning: Jinja2 placeholders found in rendered HTML")
            with open(os.path.join(OUTPUT_DIR, f'debug_rendered_{timestamp}.html'), 'w', encoding='utf-8') as f:
                f.write(html_content)
        html_path = os.path.join(OUTPUT_DIR, f'sales_forecast_report_{timestamp}.html')
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"[v{VERSION}] Generated HTML report: {html_path}")
    except Exception as e:
        error_message = f"{type(e).__name__}: {str(e)}\nStack trace:\n{''.join(traceback.format_tb(e.__traceback__))}"
        print(f"[v{VERSION}] Error rendering template: {error_message}")
        html_path = os.path.join(OUTPUT_DIR, f'sales_forecast_report_{timestamp}.html')
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sales Forecast Report</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-100">
    <div class="container mx-auto p-4">
        <div class="bg-white shadow-md rounded-lg p-6">
            <h1 class="text-3xl font-bold mb-4">Sales Forecast Report - {datetime.now().strftime('%B %d, %Y')}</h1>
            <p class="text-gray-600 mb-4">Version: {VERSION}</p>
            <p class="text-red-600 font-bold mb-4">Error: Failed to render report: {type(e).__name__}: {str(e)}</p>
            <p class="text-red-600 mb-4">Stack trace: {''.join(traceback.format_tb(e.__traceback__))}</p>
        </div>
    </div>
</body>
</html>
""")
        print(f"[v{VERSION}] Generated fallback HTML report: {html_path}")

def main():
    config = load_config()
    start_time = datetime.now()
    print(f"[v{VERSION}] Script started at {start_time}")
    timestamp = start_time.strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description="Sales Forecasting Script")
    parser.add_argument("--holdout-days", type=int, default=config['periods']['holdout_days'], help="Number of days for hold-out period")
    parser.add_argument("--forecast-days", type=int, default=config['periods']['forecast_days'], help="Number of days for forecast period")
    parser.add_argument("--force-retrain", action="store_true", help="Force full retrain")
    args = parser.parse_args()
    df = load_data()
    products = df['product_id'].unique()
    print(f"[v{VERSION}] Loaded data: {len(df)} rows, products: {products}, zero quantity ratio: {df['quantity'].eq(0).mean():.2f}")
    hold_out_end = df['date'].max()
    hold_out_start = hold_out_end - pd.Timedelta(days=args.holdout_days)
    forecast_start = hold_out_end + pd.Timedelta(days=1)
    forecast_end = forecast_start + pd.Timedelta(days=args.forecast_days)
    results = []
    daily_predictions = []
    weekly_predictions = []
    feature_insights = []
    sparse_products = []
    high_smape_products = []
    print(f"[v{VERSION}] Checking holdout data availability for {args.holdout_days} days ({hold_out_start} to {hold_out_end})")
    min_holdout_ratio = 0.3
    last_update_file = os.path.join(OUTPUT_DIR, "last_update.txt")
    last_retrain_file = os.path.join(OUTPUT_DIR, "last_retrain.txt")
    last_update = datetime.fromtimestamp(0)
    last_retrain = datetime.fromtimestamp(0)
    if os.path.exists(last_update_file):
        with open(last_update_file, 'r') as f:
            last_update = datetime.strptime(f.read().strip(), "%Y-%m-%d %H:%M:%S")
    if os.path.exists(last_retrain_file):
        with open(last_retrain_file, 'r') as f:
            last_retrain = datetime.strptime(f.read().strip(), "%Y-%m-%d %H:%M:%S")
    new_data = df[df['date'] > last_update]
    current_day = start_time.date()
    last_retrain_day = last_retrain.date()
    retrain_needed = args.force_retrain or (current_day - last_retrain_day).days >= 7
    # Add historical actuals for all products
    for product in products:
        product_df = df[df['product_id'] == product]
        historical_actuals = product_df[product_df['date'] < hold_out_start][['date', 'quantity']]
        historical_actuals = historical_actuals.drop_duplicates(subset=['date'], keep='first')
        historical_actuals['date'] = pd.to_datetime(historical_actuals['date']).dt.normalize()
        if not historical_actuals.empty:
            for _, row in historical_actuals.iterrows():
                key = (product, row['date'].strftime('%Y-%m-%d'), 'Actual', 'Historical')
                if key not in [(d['product'], d['date'], d['model'], d['period']) for d in daily_predictions]:
                    daily_predictions.append({
                        'product': product,
                        'date': row['date'].strftime('%Y-%m-%d'),
                        'model': 'Actual',
                        'period': 'Historical',
                        'actual': float(row['quantity']),
                        'predicted': None
                    })
    for product in products:
        product_df = df[df['product_id'] == product]
        holdout_df = product_df[product_df['date'].between(hold_out_start, hold_out_end)]
        print(f"[v{VERSION}] Product {product}: {len(holdout_df)} rows in holdout period")
        if len(holdout_df) < args.holdout_days * min_holdout_ratio:
            print(f"[v{VERSION}] Skipping product {product}: sparse holdout data ({len(holdout_df)}/{args.holdout_days} days)")
            sparse_products.append(product)
            product_df, _, _ = prepare_data(df, product, hold_out_start)
            future_dates = pd.date_range(start=hold_out_start, end=forecast_end, freq='D')
            fallback_forecast = pd.DataFrame({
                'ds': future_dates,
                'yhat': [product_df['quantity'].mean() if not product_df.empty else 0.0] * len(future_dates)
            })
            for model_name in ['Prophet', 'XGBoost', 'LightGBM', 'CatBoost', 'Ensemble']:
                default_metrics = get_default_result(product, 'Holdout')
                results.append(default_metrics)
                results.append(get_default_result(product, 'Forecast'))
                daily_predictions.extend(save_daily_predictions(
                    product, model_name, fallback_forecast, product_df, hold_out_start, hold_out_end, forecast_start, forecast_end, sparse_products
                ))
            holdout_actuals = product_df[product_df['date'].between(hold_out_start, hold_out_end)][['date', 'quantity']]
            holdout_actuals = holdout_actuals.drop_duplicates(subset=['date'], keep='first')
            holdout_actuals['date'] = pd.to_datetime(holdout_actuals['date']).dt.normalize()
            if holdout_actuals.empty:
                print(f"[v{VERSION}] No holdout actuals for product {product}")
                results.append(get_default_result(product, 'Holdout'))
            else:
                for _, row in holdout_actuals.iterrows():
                    key = (product, row['date'].strftime('%Y-%m-%d'), 'Actual', 'Holdout')
                    if key not in [(d['product'], d['date'], d['model'], d['period']) for d in daily_predictions]:
                        daily_predictions.append({
                            'product': product,
                            'date': row['date'].strftime('%Y-%m-%d'),
                            'model': 'Actual',
                            'period': 'Holdout',
                            'actual': float(row['quantity']),
                            'predicted': None
                        })
            weekly_predictions.extend(save_weekly_predictions(
                product, {model_name: fallback_forecast for model_name in ['Prophet', 'XGBoost', 'LightGBM', 'CatBoost', 'Ensemble']},
                hold_out_start, hold_out_end, forecast_start, forecast_end
            ))
            feature_insights.append({'product': product, 'top_features': [{'Feature': 'N/A', 'Importance': 0.0}]})
            continue
        # Non-sparse product processing
        print(f"[v{VERSION}] Processing product: {product}")
        product_df, _, _ = prepare_data(df, product, hold_out_start)
        if product_df.empty:
            print(f"[v{VERSION}] Skipping product {product} due to empty data")
            sparse_products.append(product)
            mean_quantity = product_df['quantity'].mean() if not product_df.empty else 0.0
            future_dates = pd.date_range(start=hold_out_start, end=forecast_end, freq='D')
            fallback_forecast = pd.DataFrame({
                'ds': future_dates,
                'yhat': [mean_quantity] * len(future_dates)
            })
            for model_name in ['Prophet', 'XGBoost', 'LightGBM', 'CatBoost', 'Ensemble']:
                default_metrics = get_default_result(product, 'Holdout')
                results.append(default_metrics)
                results.append(get_default_result(product, 'Forecast'))
                daily_predictions.extend(save_daily_predictions(
                    product, model_name, fallback_forecast, product_df, hold_out_start, hold_out_end, forecast_start, forecast_end, sparse_products
                ))
            weekly_predictions.extend(save_weekly_predictions(
                product, {model_name: fallback_forecast for model_name in ['Prophet', 'XGBoost', 'LightGBM', 'CatBoost', 'Ensemble']},
                hold_out_start, hold_out_end, forecast_start, forecast_end
            ))
            feature_insights.append({'product': product, 'top_features': [{'Feature': 'N/A', 'Importance': 0.0}]})
            continue
        model_files = {
            'Prophet': f'prophet_model_{product}.pkl',
            'XGBoost': f'xgb_model_{product}.pkl',
            'LightGBM': f'lgb_model_{product}.pkl',
            'CatBoost': f'cat_model_{product}.pkl'
        }
        models = {}
        predictions = {}
        for name, file in model_files.items():
            model_path = os.path.join(OUTPUT_DIR, file)
            model = None
            if os.path.exists(model_path) and not retrain_needed:
                try:
                    model = pickle.load(open(model_path, 'rb'))
                    print(f"[v{VERSION}] Loaded {name} model for {product}")
                except Exception as e:
                    print(f"[v{VERSION}] Error loading {name} model: {e}, retraining...")

            # Inside the non-sparse product loop
            if model is None or retrain_needed:
                print(f"[v{VERSION}] Training {name} model for {product}")
                if name == 'Prophet':
                    model, forecast = train_prophet(product_df, hold_out_start, hold_out_end, forecast_start, forecast_end, config, retrain_needed)
                else:
                    model, forecast = train_tree(product_df, hold_out_start, hold_out_end, forecast_start, forecast_end, model_type=name.lower(), config=config, force_retrain=retrain_needed)
                if model is not None:
                    pickle.dump(model, open(model_path, 'wb'))
                    print(f"[v{VERSION}] Saved {name} model for {product}")
                predictions[name] = forecast
            else:
                future_dates = pd.date_range(start=hold_out_start, end=forecast_end, freq='D')
                future_df = pd.DataFrame({'ds': future_dates})
                future_df = future_df.merge(product_df[['date'] + FEATURES].rename(columns={'date': 'ds'}), on='ds', how='left')
                future_regressors = estimate_future_regressors(product_df, forecast_start, forecast_end)
                last_quantity = product_df[product_df['date'] < hold_out_start]['quantity'].iloc[-1] if not product_df[product_df['date'] < hold_out_start].empty else 0.0
                recent_quantities = list(product_df[product_df['date'] < hold_out_start]['quantity'].tail(7)) if len(product_df[product_df['date'] < hold_out_start]) >= 7 else [0.0] * 7
                predictions = []
                for i in range(len(future_df)):
                    recent_quantities = prepare_future_features(future_df, i, last_quantity, recent_quantities, predictions)
                    if i >= len(pd.date_range(hold_out_start, hold_out_end)):
                        idx = i - len(pd.date_range(hold_out_start, hold_out_end))
                        if idx < len(future_regressors):
                            for col in ['campaign_intensity', 'holiday', 'season', 'zero_sale_indicator']:
                                future_df.loc[i, col] = future_regressors[col].iloc[idx]
                future_df.fillna(0, inplace=True)
                if name == 'Prophet':
                    forecast = model.predict(future_df)
                    forecast['yhat'] = np.maximum(forecast['yhat'], 0)
                    forecast = forecast[['ds', 'yhat']]
                else:
                    pred = model.predict(future_df[FEATURES])
                    pred = np.maximum(pred, 0)
                    forecast = pd.DataFrame({'ds': future_dates, 'yhat': pred})
                predictions[name] = forecast

            models[name] = model
        df_actuals = product_df[product_df['date'].between(hold_out_start, hold_out_end)][['product_id', 'date', 'quantity']].rename(columns={'date': 'ds', 'quantity': 'y'})
        print(f"[v{VERSION}] df_actuals for {product}: columns={df_actuals.columns}, rows={len(df_actuals)}, head={df_actuals.head()}")
        model_weights = {}
        for name, pred in predictions.items():
            expected_rows = len(pd.date_range(start=hold_out_start, end=forecast_end, freq='D'))
            if len(pred) != expected_rows:
                print(f"[v{VERSION}] Warning: Predictions for {name} ({product}) have {len(pred)} rows, expected {expected_rows}")
                pred = pred[pred['ds'].between(hold_out_start, forecast_end)]
            print(f"[v{VERSION}] df_actuals ds dtype: {df_actuals['ds'].dtype}, predictions ds dtype: {pred['ds'].dtype}")
            metrics = evaluate_model(pred, df_actuals, product, name, hold_out_start, hold_out_end, forecast_start, forecast_end)
            if metrics is None:
                print(f"[v{VERSION}] Skipping evaluation for {product}: No holdout actuals")
                continue
            smape = metrics['SMAPE'] if pd.notnull(metrics['SMAPE']) else 100.0
            if smape > 100:
                high_smape_products.append(product)
            model_weights[name] = 1.0 / max(smape, 1e-10)
            results.append({
                'Product': product,
                'Model': name,
                'Period': 'Holdout',
                'Avg_Quantity_Pred': pred['yhat'][pred['ds'].between(hold_out_start, hold_out_end)].mean(),
                'Avg_Quantity_Real': df_actuals['y'].mean() if not df_actuals.empty else 0.0,
                'RMSE': metrics['RMSE'],
                'R2': metrics['R2'],
                'MAE': metrics['MAE'],
                'SMAPE': metrics['SMAPE']
            })
            results.append({
                'Product': product,
                'Model': name,
                'Period': 'Forecast',
                'Avg_Quantity_Pred': pred['yhat'][pred['ds'] >= forecast_start].mean(),
                'Avg_Quantity_Real': None
            })
        total_weight = sum(model_weights.values())
        model_weights = {k: v / total_weight if total_weight > 0 else 0.25 for k, v in model_weights.items()}
        ensemble_predictions = pd.DataFrame({
            'ds': predictions['Prophet']['ds'],
            'yhat': sum(model_weights[name] * predictions[name]['yhat'] for name in predictions)
        })
        ensemble_predictions['yhat'] = np.maximum(ensemble_predictions['yhat'], 0)
        ensemble_metrics = evaluate_model(ensemble_predictions, df_actuals, product, 'Ensemble', hold_out_start, hold_out_end, forecast_start, forecast_end)
        if ensemble_metrics is not None:
            smape = ensemble_metrics['SMAPE'] if pd.notnull(ensemble_metrics['SMAPE']) else 100.0
            if smape > 100:
                high_smape_products.append(product)
            results.append({
                'Product': product,
                'Model': 'Ensemble',
                'Period': 'Holdout',
                'Avg_Quantity_Pred': ensemble_predictions['yhat'][ensemble_predictions['ds'].between(hold_out_start, hold_out_end)].mean(),
                'Avg_Quantity_Real': df_actuals['y'].mean() if not df_actuals.empty else 0.0,
                'RMSE': ensemble_metrics['RMSE'],
                'R2': ensemble_metrics['R2'],
                'MAE': ensemble_metrics['MAE'],
                'SMAPE': ensemble_metrics['SMAPE']
            })
        results.append({
            'Product': product,
            'Model': 'Ensemble',
            'Period': 'Forecast',
            'Avg_Quantity_Pred': ensemble_predictions['yhat'][ensemble_predictions['ds'] >= forecast_start].mean(),
            'Avg_Quantity_Real': None
        })
        for model_name, pred in {**predictions, 'Ensemble': ensemble_predictions}.items():
            daily_predictions.extend(save_daily_predictions(
                product, model_name, pred, product_df, hold_out_start, hold_out_end, forecast_start, forecast_end, sparse_products
            ))
        weekly_predictions.extend(save_weekly_predictions(
            product, {**predictions, 'Ensemble': ensemble_predictions},
            hold_out_start, hold_out_end, forecast_start, forecast_end
        ))
        feature_insights.append(compute_feature_importance(models, FEATURES, product, timestamp))
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(OUTPUT_DIR, f'model_comparison_{timestamp}.csv'), index=False)
    daily_df = pd.DataFrame(daily_predictions)
    daily_df.to_csv(os.path.join(OUTPUT_DIR, f'daily_predictions_{timestamp}.csv'), index=False)
    weekly_df = pd.DataFrame(weekly_predictions)
    weekly_df.to_csv(os.path.join(OUTPUT_DIR, f'weekly_predictions_{timestamp}.csv'), index=False)
    csv_data = clean_data_for_json(results_df.to_dict('records'))
    csv_data_json = json.dumps(csv_data, ensure_ascii=False, allow_nan=True)
    chart_data = clean_data_for_json(daily_predictions)
    chart_data_json = json.dumps(chart_data, ensure_ascii=False, allow_nan=True)
    validated_feature_insights = []
    for product in products:
        product_features = next((item for item in feature_insights if item.get('product') == product), None)
        if product_features and isinstance(product_features.get('top_features'), list):
            validated_feature_insights.append(product_features)
        else:
            validated_feature_insights.append({'product': product, 'top_features': [{'Feature': 'N/A', 'Importance': 0.0}]})
    feature_insights_json = json.dumps(clean_data_for_json(validated_feature_insights), ensure_ascii=False, allow_nan=True)
    holdout_table = format_df(results_df[results_df['Period'] == 'Holdout'], 'holdout')
    forecast_table = format_df(results_df[results_df['Period'] == 'Forecast'][['Product', 'Model', 'Period', 'Avg_Quantity_Pred']], 'forecast')
    template_vars = {
        'report_date': start_time.strftime('%B %d, %Y'),
        'version': VERSION,
        'alerts': generate_alerts(high_smape_products, sparse_products),
        'summary': generate_summary_txt(timestamp, hold_out_start, hold_out_end, forecast_start, forecast_end),
        'interesting_fact': generate_interesting_fact(daily_predictions, products, hold_out_start, hold_out_end),
        'business_strategies': generate_business_strategies(results, products, sparse_products),
        'hold_out_start': hold_out_start.strftime('%Y-%m-%d'),
        'hold_out_end': hold_out_end.strftime('%Y-%m-%d'),
        'forecast_start': forecast_start.strftime('%Y-%m-%d'),
        'forecast_end': forecast_end.strftime('%Y-%m-%d'),
        'holdout_table': holdout_table,
        'forecast_table': forecast_table,
        'csv_data_json': csv_data_json,
        'chart_data_json': chart_data_json,
        'feature_insights_json': feature_insights_json,
        'all_products': list(products)
    }
    generate_html_report(timestamp, template_vars)
    if not new_data.empty:
        with open(last_update_file, 'w') as f:
            f.write(start_time.strftime("%Y-%m-%d %H:%M:%S"))
    if retrain_needed:
        with open(last_retrain_file, 'w') as f:
            f.write(start_time.strftime("%Y-%m-%d %H:%M:%S"))
    return results, template_vars['business_strategies']
    
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[v{VERSION}] Fatal error in main: {type(e).__name__}: {str(e)}")
        print(f"[v{VERSION}] Stack trace:\n{''.join(traceback.format_tb(e.__traceback__))}")
        raise