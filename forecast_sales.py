import pandas as pd
import numpy as np
from prophet import Prophet
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from jinja2 import Environment, FileSystemLoader, TemplateError
import os
from datetime import datetime, timedelta
import json
import traceback
import argparse
import pickle
from sklearn.model_selection import RandomizedSearchCV
import yaml
import logging
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
MODEL_PARAMS = CONFIG['model_params']
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
        future_dates = pd.date_range(start=forecast_start, end=forecast_end, freq='D')
        return pd.DataFrame({
            'ds': future_dates,
            'campaign_intensity': [0.0] * len(future_dates),
            'holiday': [0] * len(future_dates),
            'season': [0] * len(future_dates)
        })
    df_last_year = df[df['date'] >= df['date'].max() - pd.Timedelta(days=365)].copy()
    df_last_year['month_day'] = df_last_year['date'].dt.strftime('%m-%d')
    campaign_avg = df_last_year.groupby('month_day')['campaign_intensity'].mean().reset_index()
    season_map = df_last_year.groupby('month_day')['season'].first().reset_index()
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
    future_df['campaign_intensity'] = future_df['campaign_intensity'].fillna(df_last_year['campaign_intensity'].mean() if not df_last_year.empty else 0.0)
    future_df['holiday'] = future_df['holiday'].fillna(0).astype(int)
    future_df['season'] = future_df['season'].fillna(df_last_year['season'].mode()[0] if not df_last_year.empty and not df_last_year['season'].mode().empty else 0)
    return future_df[['ds', 'campaign_intensity', 'holiday', 'season']]

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

def train_prophet(product_df, hold_out_start, hold_out_end, forecast_start, forecast_end):
    product_id = product_df['product_id'].iloc[0] if not product_df.empty else 'unknown'
    print(f"[v{VERSION}] Training Prophet model for {product_id}")
    train_df = product_df[product_df['date'] < hold_out_start][['date', 'quantity'] + PROPHET_REGRESSORS].copy()
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
    model = Prophet(**MODEL_PARAMS['prophet'])
    for col in PROPHET_REGRESSORS:
        if col in train_df.columns:
            model.add_regressor(col)
    model.fit(train_df)
    future_df = pd.DataFrame({'ds': future_dates})
    future_df['campaign_intensity'] = 0.0
    future_df['holiday'] = 0
    future_df['season'] = 0
    holdout_df = product_df[product_df['date'].between(hold_out_start, hold_out_end)][['date'] + PROPHET_REGRESSORS].rename(columns={'date': 'ds'})
    future_df = future_df.merge(holdout_df, on='ds', how='left', suffixes=('', '_holdout'))
    future_regressors = estimate_future_regressors(product_df, forecast_start, forecast_end)
    for i, row in future_df.iterrows():
        if row['ds'] >= forecast_start:
            idx = i - len(pd.date_range(hold_out_start, hold_out_end))
            if idx < len(future_regressors):
                for col in ['campaign_intensity', 'holiday', 'season']:
                    future_df.loc[i, col] = future_regressors[col].iloc[idx]
    future_df['campaign_intensity'] = future_df['campaign_intensity'].fillna(0.0)
    future_df['holiday'] = future_df['holiday'].fillna(0).astype(int)
    future_df['season'] = future_df['season'].fillna(0)
    last_train_quantity = train_df['y'].iloc[-1] if not train_df.empty else 0.0
    recent_quantities = list(train_df['y'].tail(7)) if len(train_df) >= 7 else [0.0] * 7
    predictions = []
    for i in range(len(future_df)):
        recent_quantities = prepare_future_features(future_df, i, last_train_quantity, recent_quantities, predictions)
        batch_forecast = model.predict(future_df.iloc[[i]].copy())
        pred = max(batch_forecast['yhat'].iloc[0], 0)
        predictions.append(pred)
    forecast = pd.DataFrame({
        'ds': future_dates,
        'yhat': predictions
    })
    print(f"[v{VERSION}] Prophet forecast generated for {product_id}: {len(forecast)} rows")
    return model, forecast

def train_tree_model(X_train, y_train, model_type='xgboost'):
    if model_type == 'xgboost':
        model = XGBRegressor(random_state=42)
        param_dist = MODEL_PARAMS['xgboost']
    elif model_type == 'lightgbm':
        model = LGBMRegressor(random_state=42, verbosity=-1)
        param_dist = MODEL_PARAMS['lightgbm']
    elif model_type == 'catboost':
        model = CatBoostRegressor(random_state=42, verbose=0)
        param_dist = MODEL_PARAMS['catboost']
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    n_samples = len(X_train)
    if n_samples < 2:
        return None, None
    n_splits = min(3, n_samples)
    if n_splits < 2:
        model.fit(X_train, y_train)
        return model, None
    random_search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=20,
        scoring='neg_mean_squared_error',
        cv=n_splits,
        random_state=42,
        n_jobs=-1
    )
    random_search.fit(X_train, y_train)
    return random_search.best_estimator_, random_search.best_params_

def train_tree(product_df, hold_out_start, hold_out_end, forecast_start, forecast_end, model_type):
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
    X_train = train_df[FEATURES]
    y_train = train_df['quantity']
    model, _ = train_tree_model(X_train, y_train, model_type=model_type)
    if model is None:
        print(f"[v{VERSION}] {model_type.capitalize()} model training failed for product {product_id} due to insufficient data")
        return None, pd.DataFrame({
            'ds': future_dates,
            'yhat': [0.0] * len(future_dates)
        })
    future_df = pd.DataFrame({'ds': future_dates})
    future_df = future_df.merge(product_df[['date'] + FEATURES].rename(columns={'date': 'ds'}), on='ds', how='left')
    future_regressors = estimate_future_regressors(product_df, forecast_start, forecast_end)
    for i, row in future_df.iterrows():
        if row['ds'] >= forecast_start:
            idx = i - len(pd.date_range(hold_out_start, hold_out_end))
            for col in ['campaign_intensity', 'holiday', 'season']:
                if idx < len(future_regressors):
                    future_df.loc[i, col] = future_regressors[col].iloc[idx]
    future_df.fillna(0, inplace=True)
    last_train_quantity = train_df['quantity'].iloc[-1] if not train_df.empty else 0.0
    recent_quantities = list(train_df['quantity'].tail(7)) if len(train_df) >= 7 else [0.0] * 7
    predictions = []
    for i in range(len(future_df)):
        recent_quantities = prepare_future_features(future_df, i, last_train_quantity, recent_quantities, predictions)
        pred = model.predict(future_df[FEATURES].iloc[[i]])[0]
        pred = np.maximum(pred, 0)
        predictions.append(pred)
    forecast = pd.DataFrame({
        'ds': future_dates,
        'yhat': predictions
    })
    print(f"[v{VERSION}] {model_type.capitalize()} forecast generated for {product_id}: {len(forecast)} rows")
    return model, forecast

def save_daily_predictions(product, model_name, predictions, product_df, hold_out_start, hold_out_end, forecast_start, forecast_end):
    daily_data = []
    if predictions.empty or 'ds' not in predictions.columns or 'yhat' not in predictions.columns:
        print(f"[v{VERSION}] Warning: Empty or invalid predictions for {model_name} for product {product}")
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
    if holdout_regressors.empty or 'zero_sale_indicator' not in holdout_regressors.columns:
        holdout_regressors = pd.DataFrame({'ds': predictions['ds'].unique(), 'zero_sale_indicator': 0})
    holdout_regressors = holdout_regressors.drop_duplicates(subset=['ds'], keep='first')
    future_regressors = product_df[product_df['date'] >= hold_out_end][['date', 'zero_sale_indicator']].rename(columns={'date': 'ds'})
    if future_regressors.empty:
        future_regressors = pd.DataFrame({'ds': predictions[predictions['ds'] >= forecast_start]['ds'].unique(), 'zero_sale_indicator': 0})
    future_regressors = future_regressors.drop_duplicates(subset=['ds'], keep='first')
    predictions = predictions.merge(holdout_regressors, on='ds', how='left', suffixes=('', '_holdout'))
    predictions = predictions.merge(future_regressors, on='ds', how='left', suffixes=('', '_forecast'))
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
    holdout_actuals = holdout_actuals.drop_duplicates(subset=['date'], keep='first')
    holdout_actuals = holdout_actuals.rename(columns={'date': 'ds', 'quantity': 'actual'})
    holdout_actuals['ds'] = pd.to_datetime(holdout_actuals['ds'])
    predictions = predictions.merge(holdout_actuals, on='ds', how='left')
    if holdout_actuals.empty:
        print(f"[v{VERSION}] No actuals data for product {product} in holdout period")
    seen = set()
    for _, row in predictions.iterrows():
        key = (row['Product'], row['Date'], row['Model'], row['Period'])
        if key not in seen:
            actual_value = float(row['actual']) if pd.notnull(row['actual']) else None
            if row['Model'] == 'Actual' and actual_value is None:
                print(f"[v{VERSION}] Missing actual value for product {product}, date {row['Date']}")
                continue
            daily_data.append({
                'product': row['Product'],
                'date': row['Date'],
                'model': row['Model'],
                'period': row['Period'],
                'actual': actual_value,
                'predicted': float(row['yhat'])
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



def evaluate_model(product_df, predictions, start_date, end_date, model_name, hold_out_start):
    period = 'Holdout' if start_date == hold_out_start else 'Forecast'
    product_id = product_df['product_id'].iloc[0] if not product_df.empty else 'unknown'
    
    # Validate inputs
    if predictions.empty or 'ds' not in predictions.columns or 'yhat' not in predictions.columns:
        print(f"[v{VERSION}] Warning: Invalid predictions for {model_name} for product {product_id}")
        return get_default_result(product_id, period)
    
    # Normalize dates
    predictions = predictions.copy()
    predictions['ds'] = pd.to_datetime(predictions['ds'], errors='coerce').dt.normalize()
    actuals = product_df[product_df['date'].between(start_date, end_date)][['date', 'quantity']].copy()
    actuals['date'] = pd.to_datetime(actuals['date'], errors='coerce').dt.normalize()
    
    # Log data
    print(f"[v{VERSION}] Product {product_id}: Actuals rows={len(actuals)}, Predictions rows={len(predictions)}")
    if not actuals.empty:
        print(f"[v{VERSION}] Product {product_id}: Actuals sample={actuals.head().to_dict()}")
    if not predictions.empty:
        print(f"[v{VERSION}] Product {product_id}: Predictions sample={predictions.head().to_dict()}")
    
    # Check for NaN dates
    if predictions['ds'].isna().any() or actuals['date'].isna().any():
        print(f"[v{VERSION}] Warning: NaN dates detected for product {product_id}")
        return get_default_result(product_id, period)
    
    # Filter predictions for period
    period_predictions = predictions[predictions['ds'].between(start_date, end_date)].copy()
    
    metrics = {
        'Product': product_id,
        'Model': model_name,
        'Period': period,
        'Avg_Quantity_Pred': period_predictions['yhat'].mean() if not period_predictions.empty else 0.0
    }
    
    if period == 'Holdout' and not actuals.empty:
        merged = period_predictions.merge(actuals, left_on='ds', right_on='date', how='inner')
        print(f"[v{VERSION}] Product {product_id}: Merged rows={len(merged)}")
        
        if merged.empty:
            print(f"[v{VERSION}] No matching data for {model_name} for product {product_id}")
            return get_default_result(product_id, period)
        
        if 'quantity' not in merged.columns or 'yhat' not in merged.columns:
            print(f"[v{VERSION}] Missing 'quantity' or 'yhat' in merged DataFrame for product {product_id}")
            return get_default_result(product_id, period)
        
        metrics.update({
            'RMSE': np.sqrt(mean_squared_error(merged['quantity'], merged['yhat'])),
            'R2': r2_score(merged['quantity'], merged['yhat']) if len(merged) > 1 else 0.0,
            'MAE': mean_absolute_error(merged['quantity'], merged['yhat']),
            'SMAPE': smape(merged['quantity'], merged['yhat']),
            'Avg_Quantity_Real': merged['quantity'].mean()
        })
    else:
        print(f"[v{VERSION}] No actuals data for product {product_id} in {period} period")
        return get_default_result(product_id, period)
    
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
    start_time = datetime.now()
    print(f"[v{VERSION}] Script started at {start_time}")
    timestamp = start_time.strftime("%Y%m%d_%H%M%S")
    
    parser = argparse.ArgumentParser(description="Sales Forecasting Script")
    parser.add_argument("--holdout-days", type=int, default=CONFIG['periods']['holdout_days'], help="Number of days for hold-out period")
    parser.add_argument("--forecast-days", type=int, default=CONFIG['periods']['forecast_days'], help="Number of days for forecast period")
    parser.add_argument("--force-retrain", action="store_true", help="Force full retrain")
    args = parser.parse_args()
    
    df = load_data()
    products = df['product_id'].unique()
    print(f"[v{VERSION}] Loaded data: {len(df)} rows, products: {products}, zero quantity ratio: {df['quantity'].eq(0).mean():.2f}")
    
    hold_out_end = df['date'].max()
    hold_out_start = hold_out_end - pd.Timedelta(days=args.holdout_days)
    forecast_start = hold_out_end + pd.Timedelta(days=1)
    forecast_end = forecast_start + pd.Timedelta(days=args.forecast_days)
    
    # Log holdout data availability
    print(f"[v{VERSION}] Checking holdout data availability for {args.holdout_days} days ({hold_out_start} to {hold_out_end})")
    sparse_products = []
    for product in products:
        product_df = df[df['product_id'] == product]
        holdout_df = product_df[product_df['date'].between(hold_out_start, hold_out_end)]
        print(f"[v{VERSION}] Product {product}: {len(holdout_df)} rows in holdout period")
        if len(holdout_df) < args.holdout_days * 0.5:  # Flag if less than 50% coverage
            print(f"[v{VERSION}] Warning: Product {product} has sparse holdout data ({len(holdout_df)}/{args.holdout_days} days)")
            sparse_products.append(product)
    
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
    
    results = []
    daily_predictions = []
    weekly_predictions = []
    feature_insights = []
    high_smape_products = []
    
    for product in products:
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
                results.append(get_default_result(product, 'Holdout'))
                results.append(get_default_result(product, 'Forecast'))
                daily_predictions.extend(save_daily_predictions(
                    product, model_name, fallback_forecast, product_df, hold_out_start, hold_out_end, forecast_start, forecast_end
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
            if model is None or retrain_needed:
                print(f"[v{VERSION}] Training {name} model for {product}")
                if name == 'Prophet':
                    model, forecast = train_prophet(product_df, hold_out_start, hold_out_end, forecast_start, forecast_end)
                else:
                    model, forecast = train_tree(product_df, hold_out_start, hold_out_end, forecast_start, forecast_end, model_type=name.lower())
                if model is not None:
                    pickle.dump(model, open(model_path, 'wb'))
                    print(f"[v{VERSION}] Saved {name} model for {product}")
                predictions[name] = forecast
            else:
                future_dates = pd.date_range(start=hold_out_start, end=forecast_end, freq='D')
                future_df = pd.DataFrame({'ds': future_dates})
                future_df = future_df.merge(product_df[['date'] + FEATURES].rename(columns={'date': 'ds'}), on='ds', how='left')
                future_regressors = estimate_future_regressors(product_df, forecast_start, forecast_end)
                for i, row in future_df.iterrows():
                    if row['ds'] >= forecast_start:
                        idx = i - len(pd.date_range(hold_out_start, hold_out_end))
                        if idx < len(future_regressors):
                            for col in ['campaign_intensity', 'holiday', 'season']:
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
        
        # Evaluate models only once per period
        model_weights = {}
        for name, pred in predictions.items():
            if len(pred) != len(pd.date_range(start=hold_out_start, end=forecast_end, freq='D')):
                print(f"[v{VERSION}] Warning: Predictions for {name} ({product}) have {len(pred)} rows, expected {len(pd.date_range(start=hold_out_start, end=forecast_end, freq='D'))}")
                pred = pred[pred['ds'].between(hold_out_start, forecast_end)]
            metrics = evaluate_model(product_df, pred, hold_out_start, hold_out_end, name, hold_out_start)
            smape = metrics['SMAPE'] if pd.notnull(metrics['SMAPE']) else 100.0
            if smape > 100:
                high_smape_products.append(product)
            model_weights[name] = 1.0 / max(smape, 1e-10)
            results.append(metrics)
            results.append(evaluate_model(product_df, pred, forecast_start, forecast_end, name, hold_out_start))
        
        # Ensemble predictions
        total_weight = sum(model_weights.values())
        model_weights = {k: v / total_weight if total_weight > 0 else 0.25 for k, v in model_weights.items()}
        ensemble_predictions = pd.DataFrame({
            'ds': predictions['Prophet']['ds'],
            'yhat': sum(model_weights[name] * predictions[name]['yhat'] for name in predictions)
        })
        ensemble_predictions['yhat'] = np.maximum(ensemble_predictions['yhat'], 0)
        results.append(evaluate_model(product_df, ensemble_predictions, hold_out_start, hold_out_end, 'Ensemble', hold_out_start))
        results.append(evaluate_model(product_df, ensemble_predictions, forecast_start, forecast_end, 'Ensemble', hold_out_start))
        
        # Save predictions
        for model_name, pred in {**predictions, 'Ensemble': ensemble_predictions}.items():
            daily_predictions.extend(save_daily_predictions(
                product, model_name, pred, product_df, hold_out_start, hold_out_end, forecast_start, forecast_end
            ))
        
        weekly_predictions.extend(save_weekly_predictions(
            product, {**predictions, 'Ensemble': ensemble_predictions},
            hold_out_start, hold_out_end, forecast_start, forecast_end
        ))
        
        feature_insights.append(compute_feature_importance(models, FEATURES, product, timestamp))
    
    # Save actuals for holdout period
    actual_predictions = []
    for product in products:
        product_df, _, _ = prepare_data(df, product, hold_out_start)
        holdout_actuals = product_df[product_df['date'].between(hold_out_start, hold_out_end)][['date', 'quantity']]
        holdout_actuals = holdout_actuals.drop_duplicates(subset=['date'], keep='first')
        holdout_actuals['date'] = pd.to_datetime(holdout_actuals['date']).dt.normalize()
        if holdout_actuals.empty:
            print(f"[v{VERSION}] No holdout actuals for product {product}")
            results.append(get_default_result(product, 'Holdout'))
        for _, row in holdout_actuals.iterrows():
            key = (product, row['date'].strftime('%Y-%m-%d'), 'Actual', 'Holdout')
            if key not in [(d['product'], d['date'], d['model'], d['period']) for d in daily_predictions]:
                actual_predictions.append({
                    'product': product,
                    'date': row['date'].strftime('%Y-%m-%d'),
                    'model': 'Actual',
                    'period': 'Holdout',
                    'actual': float(row['quantity']),
                    'predicted': None
                })
    daily_predictions.extend(actual_predictions)
    
    # Save outputs
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
    forecast_table = format_df(results_df[results_df['Period'] == 'Forecast'][['Product', 'Model', 'Avg_Quantity_Pred']], 'forecast')
    
    template_vars = {
        'report_date': start_time.strftime('%B %d, %Y'),
        'version': VERSION,
        'alerts': generate_alerts(high_smape_products, sparse_products),
        'summary': generate_summary_txt(timestamp, hold_out_start, hold_out_end, forecast_start, forecast_end),
        'interesting_fact': generate_interesting_fact(daily_predictions, products, hold_out_start, hold_out_end),
        'business_strategies': generate_business_strategies(results, products),
        'hold_out_start': hold_out_start.strftime('%Y-%m-%d'),
        'hold_out_end': hold_out_end.strftime('%Y-%m-%d'),
        'forecast_start': forecast_start.strftime('%Y-%m-%d'),
        'forecast_end': forecast_end.strftime('%Y-%m-%d'),
        'holdout_table': holdout_table,
        'forecast_table': forecast_table,
        'csv_data_json': csv_data_json,
        'chart_data_json': chart_data_json,
        'feature_insights_json': feature_insights_json
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