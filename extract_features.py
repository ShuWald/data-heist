"""
Extract feature importance from final model results and convert to CSV.
Creates multiple views for visualization and analysis.
"""

import os
import pandas as pd
from preprocessing import X_train, y_train, X_test, y_test, feature_names
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

insights_dir = 'Insights'
features_dir = os.path.join(insights_dir, 'features analysis')
if not os.path.exists(features_dir):
    os.makedirs(features_dir)


USE_SCALER = True  # Match final_model.py configuration
scaler = None
X_train_scaled = X_train
X_test_scaled = X_test

if USE_SCALER:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

# Train the model to get ALL feature importances
model = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
model.fit(X_train_scaled, y_train)

feature_importance = model.feature_importances_
all_features_data = []

for idx, (fname, importance) in enumerate(zip(feature_names, feature_importance)):
    all_features_data.append({
        'Rank': idx + 1,
        'Feature_Index': idx,
        'Feature_Name': fname,
        'Importance_Score': importance,
    })
    
df_all_features = pd.DataFrame(all_features_data)

# Sort by importance descending
df_all_features = df_all_features.sort_values('Importance_Score', ascending=False).reset_index(drop=True)
df_all_features['Rank'] = range(1, len(df_all_features) + 1)

# Calculate percentages and cumulative importance
total_importance = df_all_features['Importance_Score'].sum()
df_all_features['Importance_Percentage'] = (df_all_features['Importance_Score'] / total_importance * 100).round(2)
df_all_features['Cumulative_Percentage'] = df_all_features['Importance_Percentage'].cumsum().round(2)

# CSV 1: ALL features ranked by importance
df_all_features.to_csv(os.path.join(features_dir, 'all_features_importance.csv'), index=False)

# CSV 2: Top 5 features only (for focused analysis)
df_top5 = df_all_features.head(5)[['Rank', 'Feature_Name', 'Importance_Score', 'Importance_Percentage', 'Cumulative_Percentage']]
df_top5.to_csv(os.path.join(features_dir, 'top_5_features.csv'), index=False)

# CSV 3: Features by category
df_features_categorized = df_all_features.copy()

def categorize_feature(feature_name):
    """Categorize features by type based on name patterns"""
    name_lower = feature_name.lower()
    
    if any(word in name_lower for word in ['home', 'housing', 'property', 'purchase']):
        return 'Home & Housing'
    elif any(word in name_lower for word in ['auto', 'car', 'vehicle']):
        return 'Automotive'
    elif any(word in name_lower for word in ['photo', 'art', 'music', 'movie', 'entertainment', 'collector']):
        return 'Arts & Entertainment'
    elif any(word in name_lower for word in ['hunting', 'fishing', 'camping', 'outdoor', 'sports']):
        return 'Outdoor & Sports'
    elif any(word in name_lower for word in ['health', 'fitness', 'wellness', 'medical']):
        return 'Health & Wellness'
    elif any(word in name_lower for word in ['education', 'learning', 'school']):
        return 'Education'
    elif any(word in name_lower for word in ['personal', 'family', 'relationship', 'marital']):
        return 'Personal & Family'
    else:
        return 'Other'

df_features_categorized['Category'] = df_features_categorized['Feature_Name'].apply(categorize_feature)
df_features_categorized_output = df_features_categorized[['Rank', 'Feature_Name', 'Category', 'Importance_Score', 'Importance_Percentage', 'Cumulative_Percentage']]
df_features_categorized_output.to_csv(os.path.join(features_dir, 'features_by_category.csv'), index=False)

# Calculate category summaries
category_summary = df_features_categorized.groupby('Category').agg({
    'Importance_Score': ['sum', 'mean', 'count'],
    'Importance_Percentage': 'sum'
}).round(4)

# Flatten column names
category_summary.columns = ['_'.join(col).strip() for col in category_summary.columns.values]
category_summary = category_summary.reset_index()
category_summary.columns = ['Category', 'Total_Importance', 'Avg_Importance', 'Feature_Count', 'Category_Percentage']
category_summary = category_summary.sort_values('Total_Importance', ascending=False)

df_category_summary = pd.DataFrame(category_summary)
df_category_summary.to_csv(os.path.join(features_dir, 'features_category_summary.csv'), index=False)

# Console output
print("="*80)
print("FEATURE IMPORTANCE ANALYSIS: All Features (30 total)")
print("="*80)

print("\n" + "-"*80)
print("TOP 10 FEATURES:")
print("-"*80)
print(df_all_features.head(10)[['Rank', 'Feature_Name', 'Importance_Score', 'Importance_Percentage', 'Cumulative_Percentage']].to_string(index=False))

print(f"\n{'-'*80}")
print("SUMMARY STATISTICS:")
print("-"*80)
print(f"Total Features: {len(df_all_features)}")
print(f"Total Importance Score: {total_importance:.6f}")
print(f"Top Feature: {df_all_features.iloc[0]['Feature_Name']} ({df_all_features.iloc[0]['Importance_Percentage']:.2f}%)")
print(f"Top 3 Features Combined: {df_all_features.head(3)['Importance_Percentage'].sum():.2f}%")
print(f"Top 5 Features Combined: {df_all_features.head(5)['Importance_Percentage'].sum():.2f}%")
print(f"Top 10 Features Combined: {df_all_features.head(10)['Importance_Percentage'].sum():.2f}%")

print(f"\n{'-'*80}")
print("FEATURE CATEGORIES (All 30 Features):")
print("-"*80)
print(df_category_summary.to_string(index=False))

print(f"\n{'='*80}")
print("CSV FILES CREATED IN Insights/features analysis/:")
print("="*80)
print("  * all_features_importance.csv (ALL 30 features ranked)")
print("  * top_5_features.csv (focused top 5)")
print("  * features_by_category.csv (all features with category classification)")
print("  * features_category_summary.csv (aggregated by category)")
print("="*80)

