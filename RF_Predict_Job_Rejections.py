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

# Drop unnecessary columns 
columns_to_remove = ['date_rejected', 'technical_test', 'interview', 'applied']
data.drop(columns=columns_to_remove, inplace=True)

# Display data types
print(data.dtypes)
print(data.describe())

# Define date columns
date_columns = ['date_applied', 'date_sourced']

# Convert date strings to datetime objects (explicit format for consistency)
for col in date_columns:
    data[col] = pd.to_datetime(data[col], format='%Y-%m-%d', errors='coerce')

# Extract only the month and fill NaN with 0
for col in date_columns:
    data[f'{col}_month'] = data[col].dt.month.fillna(0).astype(int)

# Drop original date columns after extracting the month
data.drop(columns=date_columns, inplace=True)

# Compute the average month for each 'date_sourced_month' and 'date_applied_month' and fill missing values
month_avg = data.groupby('date_sourced_month')['date_applied_month'].mean()
data['date_applied_month'] = data.apply(
    lambda row: month_avg.get(row['date_sourced_month'], 0) if pd.isna(row['date_applied_month']) else row['date_applied_month'],
    axis=1
)
# Separate target variable before encoding
y = data.pop('rejected')  # remove 'rejected' from training data

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

# Train the model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Get feature importance from the trained model
feature_importances = rf_model.feature_importances_
feature_names = X_train.columns

# Sort features by importance
sorted_indices = np.argsort(feature_importances)[::-1]
sorted_features = [(feature_names[i], feature_importances[i]) for i in sorted_indices]

# Print feature importance in descending order
print("\nFeature Importances:")
for feature, importance in sorted_features:
    print(f"{feature}: {importance:.4f}")

# Plot feature importance
plt.figure(figsize=(12, 6))
sns.barplot(x=[feature[1] for feature in sorted_features], y=[feature[0] for feature in sorted_features])
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("Feature Importance - Random Forest")
plt.show()
