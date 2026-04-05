from .preprocessingold import X_train, y_train, X_test, y_test
from .regressionmodel import RegressionModel
from .nnmodel import NNTrainer
from functions import calculate_metrics, write_log
from .dataanalysis import error_analysis, feature_importance_analysis
import numpy as np
import torch
import os

if not os.path.exists('Logs'):
    os.makedirs('Logs')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

input_size = X_train.shape[1]
nn_trainer = NNTrainer(input_size=input_size, learning_rate=0.001, device=device)

# Convert to numpy and ensure float32 for torch
X_train_np = X_train.values if hasattr(X_train, 'values') else np.asarray(X_train)
y_train_np = y_train.values if hasattr(y_train, 'values') else np.asarray(y_train)
X_test_np = X_test.values if hasattr(X_test, 'values') else np.asarray(X_test)
y_test_np = y_test.values if hasattr(y_test, 'values') else np.asarray(y_test)

X_train_np = X_train_np.astype(np.float32)
y_train_np = y_train_np.astype(np.float32)
X_test_np = X_test_np.astype(np.float32)
y_test_np = y_test_np.astype(np.float32)

nn_trainer.fit(X_train_np, y_train_np, X_test_np, y_test_np, epochs=100, batch_size=8)

# Test neural network
nn_predictions = nn_trainer.predict(X_test_np)
nn_metrics = calculate_metrics(y_test_np, nn_predictions, model_name='Neural Network')

print(f"\n✓ Neural Network tested")
print(f"  R² = {nn_metrics['R2']:.6f}")
print(f"  MAE = {nn_metrics['MAE']:.6f}")
print(f"  RMSE = {nn_metrics['RMSE']:.6f}")


model = RegressionModel(model_type='random_forest', name='Random Forest (Best)', use_scaler=False)
model.train(X_train, y_train)
print(f"  ✓ {model.name} trained")

rf_predictions = model.predict(X_test)
rf_metrics = calculate_metrics(y_test, rf_predictions, model_name=model.name)

print(f"\n✓ Random Forest tested")
print(f"  R² = {rf_metrics['R2']:.6f}")
print(f"  MAE = {rf_metrics['MAE']:.6f}")
print(f"  RMSE = {rf_metrics['RMSE']:.6f}")


comparison = {
    'Neural Network': nn_metrics,
    'Random Forest': rf_metrics
}

print("\nPerformance Summary:")
for model_name in ['Random Forest', 'Neural Network']:
    metrics = comparison[model_name]
    print(f"\n{model_name}:")
    print(f"  R²:   {metrics['R2']:.6f}")
    print(f"  MAE:  {metrics['MAE']:.6f}")
    print(f"  RMSE: {metrics['RMSE']:.6f}")
    print(f"  MAPE: {metrics['MAPE']:.6f}")

# Determine winner
if rf_metrics['R2'] > nn_metrics['R2']:
    winner = 'Random Forest'
else:
    winner = 'Neural Network'

nn_errors = error_analysis(y_test_np, nn_predictions)
rf_errors = error_analysis(y_test, rf_predictions)

print("\nNeural Network Errors:")
print(f"  Latitude:  mean={nn_errors['ERROR_ANALYSIS']['latitude']['mean_error']:.6f}, "
      f"std={nn_errors['ERROR_ANALYSIS']['latitude']['std_error']:.6f}")
print(f"  Longitude: mean={nn_errors['ERROR_ANALYSIS']['longitude']['mean_error']:.6f}, "
      f"std={nn_errors['ERROR_ANALYSIS']['longitude']['std_error']:.6f}")

print("\nRandom Forest Errors:")
print(f"  Latitude:  mean={rf_errors['ERROR_ANALYSIS']['latitude']['mean_error']:.6f}, "
      f"std={rf_errors['ERROR_ANALYSIS']['latitude']['std_error']:.6f}")
print(f"  Longitude: mean={rf_errors['ERROR_ANALYSIS']['longitude']['mean_error']:.6f}, "
      f"std={rf_errors['ERROR_ANALYSIS']['longitude']['std_error']:.6f}")


model_log = {
    'neural_network': {
        'model_type': 'PyTorch Neural Network',
        'architecture': '30 → 128 → 64 → 2',
        'device': device,
        'metrics': nn_metrics
    },
    'nn_errors': nn_errors,
    'random_forest': {
        'model_type': 'sklearn Random Forest',
        'use_scaler': False,
        'metrics': rf_metrics
    },
    'rf_errors': rf_errors,
    'comparison': {
        'winner': winner,
        'best_r2': max(nn_metrics['R2'], rf_metrics['R2']),
        'r2_difference': abs(nn_metrics['R2'] - rf_metrics['R2']),
        'recommendation': f"Use {winner} for best performance"
    }
}

write_log(model_log, 'model_comparison_final')

print("\n✓ Results logged to Logs/model_comparison_final.txt")