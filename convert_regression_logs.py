"""
Convert regression model comparison logs to CSV format.
Extracts metrics from all 7 models (6 sklearn models + neural network) for unified comparison.
"""

import os
import pandas as pd
import yaml

insights_dir = 'Insights'
models_dir = os.path.join(insights_dir, 'models comparison')
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

logs_dir = 'Logs/Regression'

# Parse all model files
models_data = []

# First, parse the 6 sklearn models from Regression folder
for filename in os.listdir(logs_dir):
    if filename.startswith('model_') and filename.endswith('.txt') and 'comparison' not in filename:
        filepath = os.path.join(logs_dir, filename)
        
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Extract model name and settings
        model_name = filename.replace('model_', '').replace('.txt', '')
        # Properly detect scaler use from filename
        use_scaler = 'with_scaler' in model_name
        model_base = model_name.replace('_no_scaler', '').replace('_with_scaler', '')
        
        # Parse YAML-like structure
        try:
            # Extract model info block
            lines = content.split('\n')
            model_type = None
            mae = None
            mse = None
            rmse = None
            r2 = None
            mape = None
            lat_mean_error = None
            lat_max_error = None
            long_mean_error = None
            long_max_error = None
            combined_euclidean = None
            
            in_metrics = False
            in_error = False
            in_latitude = False
            in_longitude = False
            
            for i, line in enumerate(lines):
                if 'model_type:' in line:
                    model_type = line.split('model_type:')[1].strip()
                elif 'MAE:' in line and in_metrics:
                    mae = float(line.split('MAE:')[1].strip())
                elif 'MSE:' in line and in_metrics:
                    mse = float(line.split('MSE:')[1].strip())
                elif 'RMSE:' in line and in_metrics:
                    rmse = float(line.split('RMSE:')[1].strip())
                elif 'R2:' in line or 'R-squared:' in line:
                    try:
                        value = line.split(':')[1].strip()
                        r2 = float(value)
                    except:
                        pass
                elif 'MAPE:' in line and in_metrics:
                    mape = float(line.split('MAPE:')[1].strip())
                elif 'metrics:' in line:
                    in_metrics = True
                    in_error = False
                elif 'ERROR_ANALYSIS:' in line:
                    in_metrics = False
                    in_error = True
                elif 'latitude:' in line and in_error:
                    in_latitude = True
                    in_longitude = False
                elif 'longitude:' in line and in_error:
                    in_latitude = False
                    in_longitude = True
                elif 'combined_euclidean:' in line:
                    combined_euclidean = float(line.split('combined_euclidean:')[1].strip())
                    in_latitude = False
                    in_longitude = False
                elif 'mean_error:' in line and in_latitude:
                    lat_mean_error = float(line.split('mean_error:')[1].strip())
                elif 'max_error:' in line and in_latitude:
                    lat_max_error = float(line.split('max_error:')[1].strip())
                elif 'mean_error:' in line and in_longitude:
                    long_mean_error = float(line.split('mean_error:')[1].strip())
                elif 'max_error:' in line and in_longitude:
                    long_max_error = float(line.split('max_error:')[1].strip())
            
            if r2 is not None:
                models_data.append({
                    'Model': model_name,
                    'Model_Base': model_base,
                    'Model_Type': model_type,
                    'Scaler': 'Yes' if use_scaler else 'No',
                    'R2': r2,
                    'MAE': mae,
                    'MSE': mse,
                    'RMSE': rmse,
                    'MAPE': mape,
                    'Lat_Mean_Error': lat_mean_error,
                    'Lat_Max_Error': lat_max_error,
                    'Long_Mean_Error': long_mean_error,
                    'Long_Max_Error': long_max_error,
                    'Combined_Euclidean_Error': combined_euclidean,
                })
        except Exception as e:
            print(f"  [!] Error parsing {filename}: {e}")

# Parse neural network from model_comparison_final.txt
nn_file = 'Logs/model_comparison_final.txt'
if os.path.exists(nn_file):
    try:
        with open(nn_file, 'r') as f:
            content = f.read()
        
        lines = content.split('\n')
        nn_r2 = None
        nn_mae = None
        nn_mse = None
        nn_rmse = None
        nn_mape = None
        nn_lat_mean = None
        nn_lat_max = None
        nn_long_mean = None
        nn_long_max = None
        nn_combined = None
        
        in_nn_section = False
        in_nn_metrics = False
        in_nn_errors = False
        in_nn_lat = False
        in_nn_long = False
        
        for line in lines:
            if 'neural_network:' in line:
                in_nn_section = True
                in_nn_metrics = False
                in_nn_errors = False
            elif 'random_forest:' in line or 'comparison:' in line:
                in_nn_section = False
                in_nn_metrics = False
                in_nn_errors = False
            elif in_nn_section:
                if 'metrics:' in line:
                    in_nn_metrics = True
                    in_nn_errors = False
                elif 'ERROR_ANALYSIS:' in line or 'nn_errors:' in line:
                    in_nn_metrics = False
                    in_nn_errors = True
                elif 'latitude:' in line and in_nn_errors:
                    in_nn_lat = True
                    in_nn_long = False
                elif 'longitude:' in line and in_nn_errors:
                    in_nn_lat = False
                    in_nn_long = True
                elif in_nn_metrics:
                    if 'R2:' in line:
                        try:
                            nn_r2 = float(line.split('R2:')[1].strip())
                        except:
                            pass
                    elif 'MAE:' in line:
                        try:
                            nn_mae = float(line.split('MAE:')[1].strip())
                        except:
                            pass
                    elif 'MSE:' in line:
                        try:
                            nn_mse = float(line.split('MSE:')[1].strip())
                        except:
                            pass
                    elif 'RMSE:' in line:
                        try:
                            nn_rmse = float(line.split('RMSE:')[1].strip())
                        except:
                            pass
                    elif 'MAPE:' in line:
                        try:
                            nn_mape = float(line.split('MAPE:')[1].strip())
                        except:
                            pass
                elif in_nn_errors:
                    if 'combined_euclidean:' in line:
                        try:
                            nn_combined = float(line.split('combined_euclidean:')[1].strip())
                        except:
                            pass
                        in_nn_lat = False
                        in_nn_long = False
                    elif 'mean_error:' in line and in_nn_lat:
                        try:
                            nn_lat_mean = float(line.split('mean_error:')[1].strip())
                        except:
                            pass
                    elif 'max_error:' in line and in_nn_lat:
                        try:
                            nn_lat_max = float(line.split('max_error:')[1].strip())
                        except:
                            pass
                    elif 'mean_error:' in line and in_nn_long:
                        try:
                            nn_long_mean = float(line.split('mean_error:')[1].strip())
                        except:
                            pass
                    elif 'max_error:' in line and in_nn_long:
                        try:
                            nn_long_max = float(line.split('max_error:')[1].strip())
                        except:
                            pass
        
        if nn_r2 is not None:
            models_data.append({
                'Model': 'neural_network_no_scaler',
                'Model_Base': 'neural_network',
                'Model_Type': 'PyTorch Neural Network',
                'Scaler': 'No',
                'R2': nn_r2,
                'MAE': nn_mae,
                'MSE': nn_mse,
                'RMSE': nn_rmse,
                'MAPE': nn_mape,
                'Lat_Mean_Error': nn_lat_mean,
                'Lat_Max_Error': nn_lat_max,
                'Long_Mean_Error': nn_long_mean,
                'Long_Max_Error': nn_long_max,
                'Combined_Euclidean_Error': nn_combined,
            })
    except Exception as e:
        print(f"  [!] Error parsing neural network: {e}")

# Create DataFrames
df_all_models = pd.DataFrame(models_data)

# CSV 1: All 7 models with comprehensive metrics
sorted_models = df_all_models.sort_values('R2', ascending=False)
sorted_models.to_csv(os.path.join(models_dir, 'all_models_comparison.csv'), index=False)

# CSV 2: Error analysis comparison
error_data = []
for _, row in df_all_models.iterrows():
    error_data.append({
        'Model': row['Model'],
        'Model_Type': row['Model_Type'],
        'Scaler': row['Scaler'],
        'R2': row['R2'],
        'Latitude_Mean_Error_deg': row['Lat_Mean_Error'],
        'Latitude_Max_Error_deg': row['Lat_Max_Error'],
        'Longitude_Mean_Error_deg': row['Long_Mean_Error'],
        'Longitude_Max_Error_deg': row['Long_Max_Error'],
        'Combined_Error_deg': row['Combined_Euclidean_Error'],
        'Combined_Error_miles': row['Combined_Euclidean_Error'] * (69 + 55) / (2**0.5) if row['Combined_Euclidean_Error'] else None,
    })

df_errors = pd.DataFrame(error_data)
df_errors.to_csv(os.path.join(models_dir, 'error_analysis_by_model.csv'), index=False)

# Console summary
print("="*80)
print("7-MODEL COMPARISON: All Models Unified")
print("="*80)

print(f"\nModels analyzed: {len(df_all_models)}")
print(f"\nModel Types: {sorted(set(df_all_models['Model_Type'].dropna()))}")

print("\n" + "-"*80)
print("ALL MODELS RANKED BY R2:")
print("-"*80)
display_cols = ['Model', 'Model_Type', 'R2', 'MAE', 'RMSE', 'MAPE', 'Scaler']
print(sorted_models[display_cols].to_string(index=False))

print("\n" + "-"*80)
print("OVERALL STATISTICS:")
print("-"*80)
print(f"Best Model: {sorted_models.iloc[0]['Model']} (R2: {sorted_models.iloc[0]['R2']:.6f})")
print(f"Worst Model: {sorted_models.iloc[-1]['Model']} (R2: {sorted_models.iloc[-1]['R2']:.6f})")
print(f"Average R2 (No Scaler): {df_all_models[df_all_models['Scaler'] == 'No']['R2'].mean():.6f}")
print(f"Average R2 (With Scaler): {df_all_models[df_all_models['Scaler'] == 'Yes']['R2'].mean():.6f}")

print("\n" + "="*80)
print("CSV FILES CREATED IN Insights/models comparison/:")
print("="*80)
print("  * all_models_comparison.csv (all 7 models sorted by R2)")
print("="*80)
