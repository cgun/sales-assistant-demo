import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import yaml

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

def compute_feature_importance(models, features, product, timestamp):
    importance_dict = {'product': product, 'top_features': []}
    feature_imp_dict = {}  # Aggregate importance per feature
    num_valid_models = 0
    for model_name, model in models.items():
        if model_name == 'Prophet' or model is None:
            continue
        try:
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            elif hasattr(model, 'get_feature_importance'):
                importance = model.get_feature_importance()
            else:
                continue
            # Normalize importance to sum to 1 for this model
            importance_sum = sum(importance) if sum(importance) > 0 else 1.0
            importance = [imp / importance_sum for imp in importance]
            for feat, imp in zip(features, importance):
                feature_imp_dict[feat] = feature_imp_dict.get(feat, 0) + imp
            num_valid_models += 1
        except Exception as e:
            print(f"[v{VERSION}] Error computing feature importance for {model_name} on {product}: {e}")
    if feature_imp_dict and num_valid_models > 0:
        # Average across models and normalize to 1.0
        feature_importance = [(feat, imp / num_valid_models) for feat, imp in feature_imp_dict.items()]
        total_imp = sum(imp for _, imp in feature_importance)
        if total_imp > 0:
            feature_importance = [(feat, imp / total_imp) for feat, imp in feature_importance]
        feature_importance = sorted(feature_importance, key=lambda x: x[1], reverse=True)[:5]
        importance_dict['top_features'] = [
            {'Feature': feat, 'Importance': float(imp)} for feat, imp in feature_importance
        ]
    if not importance_dict['top_features']:
        importance_dict['top_features'] = [{'Feature': 'N/A', 'Importance': 0.0}]
    return importance_dict

def clean_data_for_json(data):
    if isinstance(data, list):
        return [clean_data_for_json(item) for item in data]
    elif isinstance(data, dict):
        return {key: clean_data_for_json(value) for key, value in data.items()}
    elif isinstance(data, (np.floating, float)):
        return float(data) if not np.isnan(data) else None
    elif isinstance(data, (np.integer, int)):
        return int(data)
    elif isinstance(data, pd.Timestamp):
        return data.strftime('%Y-%m-%d')
    elif pd.isna(data):
        return None
    return data

def get_default_result(product_id, period):
    print(f"[v{VERSION}] Skipping evaluation for {product_id}: No holdout actuals")
    return {
        'Product': product_id,
        'Model': 'N/A',
        'Period': period,
        'Avg_Quantity_Pred': 0.0,
        'RMSE': 0.0,
        'R2': 0.0,
        'MAE': 0.0,
        'SMAPE': 0.0,
        'Avg_Quantity_Real': 0.0
    }

def load_holidays():
    try:
        holidays = pd.read_csv('holidays.csv')
        holidays['ds'] = pd.to_datetime(holidays['ds'], errors='coerce')
        holidays = holidays.dropna(subset=['ds'])
        print(f"[v{VERSION}] Loading holidays from holidays.csv")
        return holidays
    except Exception as e:
        print(f"[v{VERSION}] Error loading holidays: {e}")
        return pd.DataFrame(columns=['ds', 'holiday'])

def generate_alerts(high_mape_products, sparse_products):
    alerts = []
    if high_mape_products:
        alerts.append(f"High MAPE detected for products: {', '.join(high_mape_products)}. Consider reviewing data quality or model parameters.")
    if sparse_products:
        alerts.append(f"Sparse data detected for products: {', '.join(sparse_products)}. Consider extending the data collection period or adjusting holdout_days.")
    if not alerts:
        alerts.append("No significant issues detected in the forecasting process.")
    return alerts

def generate_business_strategies(results, products, sparse_products=None):
    if sparse_products is None:
        sparse_products = []
    strategies = []
    results_df = pd.DataFrame(results)
    for product in products:
        if product in sparse_products:
            print(f"[v{VERSION}] Skipping strategy generation for sparse product {product}")
            strategies.append({
                'Product': product,
                'Strategy': f"No sufficient data for {product}. Consider collecting more data or reviewing holdout period."
            })
            continue
        product_results = results_df[(results_df['Product'] == product) & (results_df['Period'] == 'Holdout')]
        if product_results.empty:
            print(f"[v{VERSION}] No holdout results for product {product}, skipping strategy generation")
            strategies.append({
                'Product': product,
                'Strategy': f"No sufficient data for {product}. Consider collecting more data or reviewing holdout period."
            })
            continue
        smape_values = product_results['SMAPE']
        if smape_values.isna().all() or product_results.empty:
            print(f"[v{VERSION}] All SMAPE values are NaN for product {product}, using default strategy")
            strategies.append({
                'Product': product,
                'Strategy': f"Insufficient holdout data for {product}. Defaulting to Prophet model for forecasting."
            })
            continue
        best_model_idx = smape_values.idxmin()
        if pd.isna(best_model_idx):
            print(f"[v{VERSION}] Unable to determine best model for product {product}, using default strategy")
            strategies.append({
                'Product': product,
                'Strategy': f"Unable to determine best model for {product} due to invalid SMAPE values. Defaulting to Prophet model."
            })
            continue
        best_model = product_results.loc[best_model_idx]
        avg_pred = product_results['Avg_Quantity_Pred'].mean()
        avg_real = product_results['Avg_Quantity_Real'].mean()
        strategy = f"For product {product}, the best model is {best_model['Model']} with SMAPE {best_model['SMAPE']:.2f}%. "
        if pd.notna(best_model['SMAPE']) and best_model['SMAPE'] > 50:
            strategy += "High prediction error detected. Consider increasing stock buffer by 20% to account for uncertainty."
        elif pd.notna(avg_pred) and pd.notna(avg_real) and avg_pred > avg_real * 1.5:
            strategy += f"Demand is forecasted to increase significantly ({avg_pred:.2f} vs {avg_real:.2f}). Plan for additional inventory and promotional campaigns."
        else:
            strategy += f"Stable demand forecasted ({avg_pred:.2f}). Maintain current inventory levels and monitor closely."
        strategies.append({
            'Product': product,
            'Strategy': strategy
        })
    return strategies

def generate_interesting_fact(predictions, products, hold_out_start, hold_out_end):
    predictions_df = pd.DataFrame(predictions)
    max_diff_product = None
    max_diff = 0
    sparse_products = []
    expected_rows = len(pd.date_range(start=hold_out_start, end=hold_out_end, freq='D'))
    for product in products:
        product_preds = predictions_df[
            (predictions_df['product'] == product) &
            (predictions_df['period'] == 'Holdout') &
            (predictions_df['model'] == 'Ensemble')
        ]
        product_actuals = predictions_df[
            (predictions_df['product'] == product) &
            (predictions_df['model'] == 'Actual') &
            (predictions_df['period'] == 'Holdout')
        ]
        if product_preds.empty or product_actuals.empty:
            print(f"[v{VERSION}] Skipping product {product}: empty predictions ({len(product_preds)} rows) or actuals ({len(product_actuals)} rows)")
            sparse_products.append(product)
            continue
        if 'actual' not in product_actuals.columns:
            print(f"[v{VERSION}] No 'actual' column for product {product} in actuals, columns={list(product_actuals.columns)}")
            sparse_products.append(product)
            continue
        merged = pd.merge(
            product_preds[['date', 'predicted']],
            product_actuals[['date', 'actual']],
            on='date',
            how='inner'
        )
        if merged.empty:
            print(f"[v{VERSION}] No matching dates for product {product} after merge")
            sparse_products.append(product)
            continue
        diff = abs(merged['predicted'] - merged['actual']).mean()
        if diff > max_diff:
            max_diff = diff
            max_diff_product = product
    if max_diff_product:
        return f"Product {max_diff_product} showed the largest prediction error in the holdout period, with an average absolute difference of {max_diff:.2f} units."
    sparse_message = f"Sparse data for products: {', '.join(sparse_products)}." if sparse_products else "No data issues detected."
    return f"No significant prediction errors observed in the holdout period due to insufficient data or missing actuals. {sparse_message}"

def generate_summary_txt(timestamp, hold_out_start, hold_out_end, forecast_start, forecast_end):
    summary = (
        f"Sales Forecast Summary\n"
        f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Version: [v{VERSION}]\n"
        f"Holdout Period: {hold_out_start.strftime('%Y-%m-%d')} to {hold_out_end.strftime('%Y-%m-%d')}\n"
        f"Forecast Period: {forecast_start.strftime('%Y-%m-%d')} to {forecast_end.strftime('%Y-%m-%d')}\n"
        f"Models Used: Prophet, XGBoost, LightGBM, CatBoost, Ensemble\n"
        f"Output Files:\n"
        f"  - Model Comparison: outputs/model_comparison_{timestamp}.csv\n"
        f"  - Daily Predictions: outputs/daily_predictions_{timestamp}.csv\n"
        f"  - Weekly Predictions: outputs/weekly_predictions_{timestamp}.csv\n"
        f"  - Feature Importance: outputs/feature_importance_{timestamp}.csv\n"
        f"  - HTML Report: outputs/sales_forecast_report_{timestamp}.html\n"
    )
    os.makedirs('outputs', exist_ok=True)
    with open(f'outputs/summary_{timestamp}.txt', 'w') as f:
        f.write(summary)
    print(f"[v{VERSION}] Generated summary: outputs/summary_{timestamp}.txt")
    return summary

def format_df(df, period):
    if df.empty:
        return "<p>No data available for this period.</p>"
    df = df.copy()
    # Standardize metric columns to uppercase
    if 'smape' in df.columns:
        df['SMAPE'] = df['smape']
        df = df.drop(columns=['smape'])
    if 'rmse' in df.columns:
        df['RMSE'] = df['rmse']
        df = df.drop(columns=['rmse'])
    if 'r2' in df.columns:
        df['R2'] = df['r2']
        df = df.drop(columns=['r2'])
    if 'mae' in df.columns:
        df['MAE'] = df['mae']
        df = df.drop(columns=['mae'])
    for col in ['Avg_Quantity_Pred', 'Avg_Quantity_Real', 'RMSE', 'R2', 'MAE', 'SMAPE']:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else 'N/A')
    if period == 'holdout':
        df = df[['Product', 'Model', 'Period', 'Avg_Quantity_Pred', 'Avg_Quantity_Real', 'RMSE', 'R2', 'MAE', 'SMAPE']]
    elif period == 'forecast':
        df = df[['Product', 'Model', 'Period', 'Avg_Quantity_Pred']]
    html = "<table class='table-auto w-full border-collapse border border-gray-300'>\n<thead>\n<tr>"
    for col in df.columns:
        html += f"<th class='border border-gray-300 px-4 py-2'>{col}</th>"
    html += "</tr>\n</thead>\n<tbody>"
    for _, row in df.iterrows():
        html += "\n<tr>"
        for val in row:
            html += f"<td class='border border-gray-300 px-4 py-2'>{val}</td>"
        html += "</tr>"
    html += "\n</tbody>\n</table>"
    return html