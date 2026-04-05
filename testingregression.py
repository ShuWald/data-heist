from model import RegressionModel
from preprocessing import X_train, y_train, X_test, y_test


model1 = RegressionModel(model_type='linear_no_scaler', name='Linear (No Scaler)')
model2 = RegressionModel(model_type='linear_with_scaler', name='Linear (With Scaler)')
model3 = RegressionModel(model_type='random_forest', name='Random Forest (With Scaler)')
all_models = [model1, model2, model3]

print("Training all models...")
for model in all_models:
    model.train(X_train, y_train)
    print(f"  ✓ {model.name}")

print("\nMaking predictions...")
for model in all_models:
    predictions = model.predict(X_test)
    print(f"\n{model.name}:")
    print(f"  Predictions shape: {predictions.shape}")
    print(f"  First 5 predictions:\n{predictions[:5]}")
