from preprocessing import X_train, y_train, X_test, y_test, feature_names, address_label_encoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import os

insights_dir = 'Insights'
variance_dir = os.path.join(insights_dir, '2 vs 3 output models')
if not os.path.exists(variance_dir):
    os.makedirs(variance_dir)

# ===== CONFIGURATION =====
USE_SCALER = True  # Match final_model.py configuration
# ======================

print("="*80)
print("MODEL VARIANCE TEST: 3-Output vs 2-Output Random Forest (10 runs)")
print("="*80)
print(f"Configuration: USE_SCALER = {USE_SCALER}")
print(f"Saving results to {variance_dir}/ directory...\n")

# Initialize scaler if enabled
scaler = None
X_train_scaled = X_train
X_test_scaled = X_test

if USE_SCALER:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

# Prepare data
y_train_2output = y_train.iloc[:, :2] if hasattr(y_train, 'iloc') else y_train[:, :2]
y_test_2output = y_test.iloc[:, :2] if hasattr(y_test, 'iloc') else y_test[:, :2]

results = []

print("Training models with different random seeds...")

for seed in range(10):
    # 3-Output Model
    model_3 = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=seed, n_jobs=-1)
    model_3.fit(X_train_scaled, y_train)
    pred_3 = model_3.predict(X_test_scaled)
    
    # Get Lat/Long R² for 3-output
    lat_r2_3 = r2_score(y_test.iloc[:, 0], pred_3[:, 0])
    long_r2_3 = r2_score(y_test.iloc[:, 1], pred_3[:, 1])
    avg_r2_3 = (lat_r2_3 + long_r2_3) / 2
    
    # 2-Output Model
    model_2 = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=seed, n_jobs=-1)
    model_2.fit(X_train_scaled, y_train_2output)
    pred_2 = model_2.predict(X_test_scaled)
    
    # Get Lat/Long R² for 2-output
    lat_r2_2 = r2_score(y_test_2output.iloc[:, 0], pred_2[:, 0])
    long_r2_2 = r2_score(y_test_2output.iloc[:, 1], pred_2[:, 1])
    avg_r2_2 = (lat_r2_2 + long_r2_2) / 2
    
    # Store results
    results.append({
        'seed': seed,
        'model_3_lat_r2': lat_r2_3,
        'model_3_long_r2': long_r2_3,
        'model_3_avg_r2': avg_r2_3,
        'model_2_lat_r2': lat_r2_2,
        'model_2_long_r2': long_r2_2,
        'model_2_avg_r2': avg_r2_2,
        'lat_improvement': lat_r2_3 - lat_r2_2,
        'long_improvement': long_r2_3 - long_r2_2,
        'avg_improvement': avg_r2_3 - avg_r2_2,
        'lat_improvement_pct': ((lat_r2_3 - lat_r2_2) / lat_r2_2 * 100) if lat_r2_2 != 0 else 0,
        'long_improvement_pct': ((long_r2_3 - long_r2_2) / long_r2_2 * 100) if long_r2_2 != 0 else 0,
        'avg_improvement_pct': ((avg_r2_3 - avg_r2_2) / avg_r2_2 * 100) if avg_r2_2 != 0 else 0,
    })
    print(f"  Run {seed+1:2d}/10 complete")

# Create comprehensive DataFrame
df_runs = pd.DataFrame(results)

# CSV 1: Run-by-run comparison (detailed)
df_runs.to_csv(os.path.join(variance_dir, 'model_variance_runs.csv'), index=False)

# CSV 2: Summary statistics
summary_data = {
    'Metric': [
        'Latitude R² (3-Output) Mean',
        'Latitude R² (3-Output) Std',
        'Latitude R² (3-Output) Min',
        'Latitude R² (3-Output) Max',
        'Latitude R² (2-Output) Mean',
        'Latitude R² (2-Output) Std',
        'Latitude R² (2-Output) Min',
        'Latitude R² (2-Output) Max',
        'Longitude R² (3-Output) Mean',
        'Longitude R² (3-Output) Std',
        'Longitude R² (3-Output) Min',
        'Longitude R² (3-Output) Max',
        'Longitude R² (2-Output) Mean',
        'Longitude R² (2-Output) Std',
        'Longitude R² (2-Output) Min',
        'Longitude R² (2-Output) Max',
        'Average R² (3-Output) Mean',
        'Average R² (3-Output) Std',
        'Average R² (2-Output) Mean',
        'Average R² (2-Output) Std',
        'Latitude Improvement Mean (+)',
        'Latitude Improvement %',
        'Longitude Improvement Mean (+)',
        'Longitude Improvement %',
        'Average Improvement Mean (+)',
        'Average Improvement %',
    ],
    'Value': [
        df_runs['model_3_lat_r2'].mean(),
        df_runs['model_3_lat_r2'].std(),
        df_runs['model_3_lat_r2'].min(),
        df_runs['model_3_lat_r2'].max(),
        df_runs['model_2_lat_r2'].mean(),
        df_runs['model_2_lat_r2'].std(),
        df_runs['model_2_lat_r2'].min(),
        df_runs['model_2_lat_r2'].max(),
        df_runs['model_3_long_r2'].mean(),
        df_runs['model_3_long_r2'].std(),
        df_runs['model_3_long_r2'].min(),
        df_runs['model_3_long_r2'].max(),
        df_runs['model_2_long_r2'].mean(),
        df_runs['model_2_long_r2'].std(),
        df_runs['model_2_long_r2'].min(),
        df_runs['model_2_long_r2'].max(),
        df_runs['model_3_avg_r2'].mean(),
        df_runs['model_3_avg_r2'].std(),
        df_runs['model_2_avg_r2'].mean(),
        df_runs['model_2_avg_r2'].std(),
        df_runs['lat_improvement'].mean(),
        df_runs['lat_improvement_pct'].mean(),
        df_runs['long_improvement'].mean(),
        df_runs['long_improvement_pct'].mean(),
        df_runs['avg_improvement'].mean(),
        df_runs['avg_improvement_pct'].mean(),
    ]
}

df_summary = pd.DataFrame(summary_data)
df_summary.to_csv(os.path.join(variance_dir, 'model_variance_summary.csv'), index=False)

# CSV 3: Improvement-focused (for easy charting)
df_improvements = df_runs[['seed', 'lat_improvement', 'long_improvement', 'avg_improvement', 
                            'lat_improvement_pct', 'long_improvement_pct', 'avg_improvement_pct']].copy()
df_improvements.columns = ['Run_Seed', 'Latitude_Improvement', 'Longitude_Improvement', 'Average_Improvement',
                           'Latitude_Improvement_Pct', 'Longitude_Improvement_Pct', 'Average_Improvement_Pct']
df_improvements.to_csv(os.path.join(variance_dir, 'model_improvements.csv'), index=False)

# Console summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

lat_diff = df_runs['model_3_lat_r2'].mean() - df_runs['model_2_lat_r2'].mean()
long_diff = df_runs['model_3_long_r2'].mean() - df_runs['model_2_long_r2'].mean()
avg_diff = df_runs['model_3_avg_r2'].mean() - df_runs['model_2_avg_r2'].mean()

lat_diff_pct = df_runs['lat_improvement_pct'].mean()
long_diff_pct = df_runs['long_improvement_pct'].mean()
avg_diff_pct = df_runs['avg_improvement_pct'].mean()

print(f"\n3-OUTPUT MODEL (Lat + Long + Address):")
print(f"  Latitude R²:   {df_runs['model_3_lat_r2'].mean():.6f} ± {df_runs['model_3_lat_r2'].std():.6f}")
print(f"  Longitude R²:  {df_runs['model_3_long_r2'].mean():.6f} ± {df_runs['model_3_long_r2'].std():.6f}")
print(f"  Average R²:    {df_runs['model_3_avg_r2'].mean():.6f} ± {df_runs['model_3_avg_r2'].std():.6f}")

print(f"\n2-OUTPUT MODEL (Lat + Long only):")
print(f"  Latitude R²:   {df_runs['model_2_lat_r2'].mean():.6f} ± {df_runs['model_2_lat_r2'].std():.6f}")
print(f"  Longitude R²:  {df_runs['model_2_long_r2'].mean():.6f} ± {df_runs['model_2_long_r2'].std():.6f}")
print(f"  Average R²:    {df_runs['model_2_avg_r2'].mean():.6f} ± {df_runs['model_2_avg_r2'].std():.6f}")

print(f"\n3-OUTPUT ADVANTAGE:")
print(f"  Latitude:      {lat_diff:+.6f} ({lat_diff_pct:+.2f}%)")
print(f"  Longitude:     {long_diff:+.6f} ({long_diff_pct:+.2f}%)")
print(f"  Average:       {avg_diff:+.6f} ({avg_diff_pct:+.2f}%)")

# Statistical significance
combined_std = np.sqrt(df_runs['model_3_avg_r2'].std()**2 + df_runs['model_2_avg_r2'].std()**2)
noise_ratio = abs(avg_diff) / combined_std if combined_std > 0 else 0

print(f"\nSTATISTICAL SIGNIFICANCE:")
print(f"  Noise floor (combined std): {combined_std:.6f}")
print(f"  Difference/Noise ratio:     {noise_ratio:.2f}x")
if noise_ratio > 2:
    print(f"  [OK] HIGHLY SIGNIFICANT (>2x noise)")
elif noise_ratio > 1:
    print(f"  [OK] SIGNIFICANT (>1x noise)")
else:
    print(f"  [!] Within noise (likely random variance)")

print(f"\nCSV FILES CREATED:")
print(f"  * {insights_dir}/model_variance_runs.csv")
print(f"  * {insights_dir}/model_variance_summary.csv")
print(f"  * {insights_dir}/model_improvements.csv")
print("="*80)
