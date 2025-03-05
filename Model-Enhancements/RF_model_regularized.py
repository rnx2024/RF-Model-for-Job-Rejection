import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from category_encoders import TargetEncoder

# Load dataset
data = pd.read_csv('cleaned_data.csv')

# Remove unnecessary columns 
columns_to_remove = ['technical_test', 'interview', 'applied', 'date_rejected']
data.drop(columns=columns_to_remove, inplace=True)

# Define date columns
date_columns = ['date_applied', 'date_sourced']

# Convert date strings to datetime objects 
for col in date_columns:
    data[col] = pd.to_datetime(data[col], format='%Y-%m-%d', errors='coerce')

# Extract only the month and fill NaN with 0
for col in date_columns:
    data[f'{col}_month'] = data[col].dt.month.fillna(0).astype(int)

# Drop original date columns after extracting the month
data.drop(columns=date_columns, inplace=True)

# Compute average dates for 'date_applied_month" and to replace missing values
month_avg = data.groupby('date_sourced_month')['date_applied_month'].mean() #use date_sourced_month to group date_applied_month
data['date_applied_month'] = data.apply(
    lambda row: month_avg.get(row['date_sourced_month'], 0) if pd.isna(row['date_applied_month']) else row['date_applied_month'],
    axis=1
)

# Separate target variable before encoding
y = data.pop('rejected')  # Ensure 'rejected' is removed from training data

# Identify categorical columns
categorical_cols = ['company_name', 'position', 'job_description', 'location',
                    'contract_type', 'language', 'job_location', 'mode_of_application',
                    'job_delisted']

# Apply target encoding
encoder = TargetEncoder(cols=categorical_cols)
X_encoded = encoder.fit_transform(data, y)  # Encoding categorical features

# Ensure all features are numeric and handle NaN values
X_encoded = X_encoded.apply(pd.to_numeric, errors='coerce').fillna(0)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Define and configure the RandomForestRegressor with regularization
rf_model = RandomForestRegressor(
    n_estimators=100,  # Keep the number of trees to 100
    max_depth=10,  # Limit the depth of each tree to prevent overfitting
    min_samples_split=5,  # Require at least 5 samples to split a node
    min_samples_leaf=2,  # Ensure at least 2 samples exist in each leaf node
    max_features='sqrt',  # Limit the number of features considered at each split
    random_state=42  
)

# Train the model on the training data
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(X_test)

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error with Regularization: {mse}')

# Get feature importance from the trained model
feature_importances = rf_model.feature_importances_
feature_names = X_train.columns

# Sort features by importance
sorted_indices = np.argsort(feature_importances)[::-1]
sorted_features = [(feature_names[i], feature_importances[i]) for i in sorted_indices]

# Print feature importance in descending order
print("\nFeature Importances (with Regularization):")
for feature, importance in sorted_features:
    print(f"{feature}: {importance:.4f}")

# Plot feature importance
plt.figure(figsize=(12, 6))
sns.barplot(x=[feature[1] for feature in sorted_features], y=[feature[0] for feature in sorted_features])
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("Feature Importance - Random Forest (With Regularization)")
plt.show()
