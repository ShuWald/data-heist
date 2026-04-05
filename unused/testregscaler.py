"""
Scaler Testing File - Archived for Reference
This file tests the impact of using StandardScaler on our regression models.
Results showed scaler DECREASED performance across all models.
See observations.txt for detailed analysis.
"""

from testing.regressionmodel import RegressionModel
from testing.preprocessingold import X_train, y_train, X_test, y_test
from functions import calculate_metrics, write_log
from testing.dataanalysis import (
    analyze_output_range,
    correlation_analysis,
    data_sufficiency_check,
    error_analysis,
    feature_importance_analysis,
    compare_models,
    learning_curves
)
import os

data_log = {}
data_log.update(analyze_output_range(y_train))
data_log.update(correlation_analysis(X_train, y_train))
data_log.update(data_sufficiency_check(X_train, X_test))
print("✓ Data analysis complete")

model4 = RegressionModel(model_type='linear', name='Linear (With Scaler)', use_scaler=True)
model5 = RegressionModel(model_type='ridge', name='Ridge (With Scaler)', use_scaler=True)
model6 = RegressionModel(model_type='random_forest', name='Random Forest (With Scaler)', use_scaler=True)

all_models_scaled = [model4, model5, model6]

for model in all_models_scaled:
    model.train(X_train, y_train)
    print(f"  ✓ {model.name} trained")

model_metrics_scaled = {}
model_predictions_scaled = {}
model_logs_scaled = {}

for model in all_models_scaled:
    predictions = model.predict(X_test)
    metrics = calculate_metrics(y_test, predictions, model_name=model.name)
    model_metrics_scaled[model.name] = metrics
    model_predictions_scaled[model.name] = predictions

# Print comparison to console
print("\nModel Comparison:")
comparison_log_scaled = compare_models(model_metrics_scaled)
for model_name, r2 in sorted([(m, model_metrics_scaled[m]['R2']) for m in model_metrics_scaled], key=lambda x: x[1], reverse=True):
    print(f"  {model_name}: R² = {r2:.6f}")


for model in all_models_scaled:
    errors = error_analysis(y_test, model_predictions_scaled[model.name])
    
    # Create individual model log
    model_log = {
        'model_name': model.name,
        'model_type': model.model_type,
        'use_scaler': model.use_scaler,
        'metrics': model_metrics_scaled[model.name]
    }
    model_log.update(errors)
    model_log.update(feature_importance_analysis(model))
    
    model_logs_scaled[model.name] = model_log
    print(f"  ✓ {model.name} analyzed")

# Create Logs directory path in unused folder
logs_dir = os.path.join('Logs', 'Regression')
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

# Write data analysis log
print("\nData Analysis Log:")
write_log(data_log, os.path.join(logs_dir, 'data_analysis_with_scaler'))
comparison_log_scaled.update(data_log)
write_log(comparison_log_scaled, os.path.join(logs_dir, 'model_comparison_with_scaler'))

# Write individual model logs
print("\nModel Logs:")
for model_name, model_log in model_logs_scaled.items():
    filename = f"model_{model_name.replace(' ', '_').replace('(', '').replace(')', '').lower()}"
    write_log(model_log, os.path.join(logs_dir, filename))
