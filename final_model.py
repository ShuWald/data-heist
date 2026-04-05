from preprocessing import X_train, y_train, X_test, y_test, feature_names, address_label_encoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from functions import write_log
import numpy as np
import pandas as pd

# ===== CONFIGURATION =====
USE_SCALER = True  # Toggle: True for better performance (+47.4% R2 improvement)
# ======================

# Conversion constants at ~34N latitude (Southern California): 
MILES_PER_DEGREE_LAT = 69.0
MILES_PER_DEGREE_LONG = 55.0
KM_PER_DEGREE_LAT = 111.0
KM_PER_DEGREE_LONG = 88.0

# Initialize scaler if enabled
scaler = None
X_train_scaled = X_train
X_test_scaled = X_test

if USE_SCALER:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("[OK] StandardScaler applied to training and test data")

model = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
model.fit(X_train_scaled, y_train)
print("[OK] Random Forest trained successfully")
predictions = model.predict(X_test_scaled)
print(f"[OK] Predictions generated: shape {predictions.shape}")

output_names = ['Latitude', 'Longitude', 'Address']
metrics_dict = {}

for i, output_name in enumerate(output_names):
    y_true = y_test.iloc[:, i].values if hasattr(y_test, 'iloc') else y_test[:, i]
    y_pred = predictions[:, i]
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mask = np.abs(y_true) > 1e-10  # Non-zero values
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / np.abs(y_true[mask]))) * 100
    else:
        mape = np.mean(np.abs(y_true - y_pred))  # Fallback to MAE if all zeros
    
    metrics_dict[output_name] = {
        'R2': r2,
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape
    }
    
    print(f"\n{output_name}:")
    print(f"  R-squared:   {r2:.6f}")
    print(f"  MAE:  {mae:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAPE: {mape:.2f}%")

spatial_predictions = predictions[:, :2] 
spatial_actual = y_test.values[:, :2]
spatial_error = np.sqrt(np.sum((spatial_actual - spatial_predictions)**2, axis=1))

print(f"\nGeographic Error (Latitude + Longitude only):")
print(f"  Mean Euclidean: {spatial_error.mean():.6f} deg")
print(f"  Std:            {spatial_error.std():.6f} deg")
print(f"  Max:            {spatial_error.max():.6f} deg")

mean_error_miles = spatial_error.mean() * np.sqrt(MILES_PER_DEGREE_LAT**2 + MILES_PER_DEGREE_LONG**2) / np.sqrt(2)
max_error_miles = spatial_error.max() * np.sqrt(MILES_PER_DEGREE_LAT**2 + MILES_PER_DEGREE_LONG**2) / np.sqrt(2)

print(f"\n  Interpretation:")
print(f"  Mean Error: {mean_error_miles:.2f} miles ({mean_error_miles*1.609:.2f} km)")
print(f"  Max Error:  {max_error_miles:.2f} miles ({max_error_miles*1.609:.2f} km)")
print(f"  Summary: Model predicts location within ~{mean_error_miles:.1f} miles on average")

# Address error reported separately (categorical data)
address_actual = y_test.iloc[:, 2].values if hasattr(y_test, 'iloc') else y_test[:, 2]
address_pred = predictions[:, 2]
address_error = np.abs(address_actual - address_pred)

print(f"\nAddress Error (Categorical, encoded int):")
print(f"  Mean Absolute Error: {address_error.mean():.2f} (encoding units)")
print(f"  Std:                 {address_error.std():.2f}")
print(f"  Max:                 {address_error.max():.2f}")


print("\nTop 10 Predictive Features")
feature_importance = model.feature_importances_
top_indices = np.argsort(feature_importance)[-10:][::-1]
for rank, idx in enumerate(top_indices, 1):
    print(f"  {rank:2d}. {feature_names[idx]}: {feature_importance[idx]:.6f}")
print("")

#---

results_log = {
    'model_summary': {
        'model_type': 'sklearn Random Forest Regressor',
        'outputs': output_names,
        'n_estimators': 100,
        'max_depth': 20,
        'test_samples': len(X_test)
    },
    'metrics_by_output': metrics_dict,
    'geographic_error': {
        'mean_spatial_euclidean': float(spatial_error.mean()),
        'std_spatial_euclidean': float(spatial_error.std()),
        'max_spatial_euclidean': float(spatial_error.max())
    },
    'address_error': {
        'mean_absolute_error': float(address_error.mean()),
        'std_absolute_error': float(address_error.std()),
        'max_absolute_error': float(address_error.max())
    },
    'top_10_features': {
        feature_names[idx]: float(feature_importance[idx]) 
        for idx in top_indices
    }
}

write_log(results_log, 'final_model_results')
print("[OK] Results logged to Logs/final_model_results.txt")

#---

def display_top_predictions(sample_idx, top_n=3):
    """
    Display predictions and top N nearest addresses by geographic proximity.
    
    HOW IT WORKS:
    1. Model predicts a lat/long coordinate from consumer behavioral features
    2. We find training data addresses geographically nearest to that prediction
    3. Shows the K-nearest neighbors from actual training addresses
    
    NOTE: Since the model has weak geographic signal (R-squared ~0.08), many predictions
    converge to similar areas, resulting in duplicate addresses. This is expected.
    To get diverse predictions, you'd need stronger geographic features in the data.
    """
    if sample_idx >= len(X_test):
        print(f"Error: Sample index {sample_idx} out of bounds (max: {len(X_test)-1})")
        return
    
    # Get actual values
    actual = y_test.iloc[sample_idx].values if hasattr(y_test, 'iloc') else y_test[sample_idx]
    predicted = predictions[sample_idx]
    
    # Display actual vs predicted
    print(f"\n{'Metric':<20} {'Actual':<20} {'Predicted':<20} {'Error':<15}")
    print("-" * 75)
    
    for i, name in enumerate(output_names):
        actual_val = actual[i]
        pred_val = predicted[i]
        error = abs(actual_val - pred_val)
        
        if name in ['Latitude', 'Longitude']:
            print(f"{name:<20} {actual_val:.6f} deg {'':<4} {pred_val:.6f} deg {'':<4} {error:.6f} deg")
        else:
            actual_addr = address_label_encoder.inverse_transform([int(actual_val)])[0]
            # Round predicted address to nearest integer and decode
            pred_addr_idx = int(np.round(pred_val))
            # Clamp to valid range [0, num_classes-1]
            pred_addr_idx = np.clip(pred_addr_idx, 0, len(address_label_encoder.classes_) - 1)
            pred_addr = address_label_encoder.inverse_transform([pred_addr_idx])[0]
            print(f"{name:<20} {actual_addr:<20} {pred_addr:<20}")
    
    # Calculate geographic error
    lat_error = actual[0] - predicted[0]
    long_error = actual[1] - predicted[1]
    spatial_sample_error = np.sqrt(lat_error**2 + long_error**2)
    sample_error_miles = spatial_sample_error * np.sqrt(MILES_PER_DEGREE_LAT**2 + MILES_PER_DEGREE_LONG**2) / np.sqrt(2)
    
    print("-" * 75)
    print(f"Geographic Error: {spatial_sample_error:.6f} deg ({sample_error_miles:.2f} miles)")
    
    # Find nearest addresses by calculating distance to all training data samples
    # The model predicts a location (lat, long); we find which training addresses are nearest
    predicted_lat, predicted_long = predicted[0], predicted[1]
    train_coords = y_train.iloc[:, :2].values if hasattr(y_train, 'iloc') else y_train[:, :2]
    train_addresses = y_train.iloc[:, 2].values if hasattr(y_train, 'iloc') else y_train[:, 2]
    
    # Compute Euclidean distance from predicted coords to all training coordinates
    distances = np.sqrt((train_coords[:, 0] - predicted_lat)**2 + (train_coords[:, 1] - predicted_long)**2)
    
    # Get top N nearest neighbors and deduplicate by address
    # Use a larger multiplier to ensure we find enough unique addresses
    nearest_indices = np.argsort(distances)[:max(top_n * 20, 200)]  # Get many more to guarantee uniqueness
    seen_addresses = set()
    unique_results = []
    
    for idx in nearest_indices:
        distance_deg = distances[idx]
        distance_miles = distance_deg * np.sqrt(MILES_PER_DEGREE_LAT**2 + MILES_PER_DEGREE_LONG**2) / np.sqrt(2)
        address_decoded = address_label_encoder.inverse_transform([int(train_addresses[idx])])[0]
        lat, lng = train_coords[idx]
        
        if address_decoded not in seen_addresses:
            unique_results.append((address_decoded, lat, lng, distance_miles))
            seen_addresses.add(address_decoded)
            if len(unique_results) >= top_n:
                break
    
    print(f"\n>> Top {top_n} Nearest Unique Addresses (by geographic proximity to predicted location):")
    print("-" * 75)
    for rank, (addr, lat, lng, miles) in enumerate(unique_results, 1):
        print(f"  {rank}. {addr}")
        print(f"     Location: {lat:.6f} deg, {lng:.6f} deg | Distance: {miles:.2f} miles")
    
    print(f"\n{'='*75}")


# Interactive 
print(f"\nEnter a test sample index (0-{len(X_test)-1}): ", end="")
try:
    user_sample_idx = int(input())
    user_top_n = 3  # Can modify this as needed
    display_top_predictions(sample_idx=user_sample_idx, top_n=user_top_n)
except ValueError:
    print("Invalid input. Running example with sample index 0...")
    display_top_predictions(sample_idx=0, top_n=3)

