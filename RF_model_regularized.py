import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, confusion_matrix
from category_encoders import TargetEncoder

# Define log infromation format
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

<<<<<<< HEAD
def load_dataset(file_path):
    """
    Load the dataset from a CSV file.
    """
    try:
        data = pd.read_csv(file_path)
        logging.info(f"Dataset loaded from {file_path}.")
        return data
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}. Please check the file path and retry.")
        raise
    except pd.errors.ParserError:
        logging.error(f"Error parsing the file at {file_path}. Please check if the file is properly formatted and retry.")

        raise
=======
# Remove unnecessary column
columns_to_remove = ['date_rejected']
data.drop(columns=columns_to_remove, inplace=True)

# Define date columns
date_columns = ['date_applied', 'date_sourced']

# Convert date strings to datetime objects 
for col in date_columns:
    data[col] = pd.to_datetime(data[col], format='%Y-%m-%d', errors='coerce')
>>>>>>> 2bfa1e1eccde5f892f41448175222fad7922d6e0

def preprocess_data(data, date_columns, drop_column):
    """
    Preprocess the dataset: remove the date_rejected column, convert date strings to datetime objects
    and extract the month from date columns.

    Returns:
        pd.DataFrame: Preprocessed dataset.
    """
    try:
        data.drop(columns=drop_column, inplace=True)
        logging.info(f"Removed column:{drop_column}.")

<<<<<<< HEAD
        for col in date_columns:
            data[col] = pd.to_datetime(data[col], format='%Y-%m-%d', errors='raise') # Convert to datetime
            logging.info(f"Converted column {col} to datetime.")

        for col in date_columns:
            data[f'{col}_month'] = data[col].dt.month.fillna(0).astype(int) # Extract month
            logging.info(f"Extracted month from column {col}.")

        data.drop(columns=date_columns, inplace=True)
        logging.info(f"Dropped original date columns after extracting month: {date_columns}.")
=======
# Compute average dates for 'date_applied_month" and to replace missing values
month_avg = data.groupby('date_sourced_month')['date_applied_month'].mean() #use date_sourced_month to group date_applied_month
data['date_applied_month'] = data.apply(
    lambda row: month_avg.get(row['date_sourced_month'], 0) if pd.isna(row['date_applied_month']) else row['date_applied_month'],
    axis=1
)
>>>>>>> 2bfa1e1eccde5f892f41448175222fad7922d6e0

        month_avg = data.groupby('date_sourced_month')['date_applied_month'].mean() # Calculate average of 'date_applied_month' for each 'date_sourced_month'
        data['date_applied_month'] = data.apply(
            lambda row: month_avg.get(row['date_sourced_month'], 0) if pd.isna(row['date_applied_month']) else row['date_applied_month'],
            axis=1
        )
        logging.info("Filled missing values in 'date_applied_month' using the average of 'date_sourced_month'.")
        
        return data
    except KeyError as e:
        logging.error(f"KeyError in preprocessing data: {e}. Ensure the column names are correct.")
        raise
    except pd.errors.ParserError as e:
        logging.error(f"ParserError in preprocessing data: {e}. Check the date format in the dataset.")
        raise
    except Exception as e:
        logging.error(f"Unexpected error in preprocessing data: {e}.")
        raise

<<<<<<< HEAD
def encode_features(data, target, categorical_cols):
    """
    Apply target encoding to categorical features.
=======
# Identify categorical columns
categorical_cols = ['company_name', 'position', 'job_description', 'location','technical_test', 'interview', 'applied',
                    'contract_type', 'language', 'job_location', 'mode_of_application','job_delisted']
>>>>>>> 2bfa1e1eccde5f892f41448175222fad7922d6e0

    Returns:
        pd.DataFrame: Target encoded dataset.
    """
    try:
        encoder = TargetEncoder(cols=categorical_cols)
        x_encoded = encoder.fit_transform(data, target)
        x_encoded.columns = [col + '_encoded' if col in categorical_cols else col for col in x_encoded.columns]
        logging.info(f"Applied target encoding to columns: {categorical_cols}.")
        return x_encoded
    except Exception as e:
        logging.error(f"Error in encoding features with TargetEncoder: {e}. Please ensure the categorical columns are correct.")
        raise

def train_random_forest(X_train, y_train, **kwargs):
    """
    Configure and train the Random Forest model

    Returns:
        RandomForestRegressor: Trained Random Forest model.
    """
    try:
        rf_model = RandomForestRegressor(random_state=42, oob_score=True, **kwargs)
        rf_model.fit(X_train, y_train)
        logging.info("Trained Random Forest model.")
        return rf_model
    except Exception as e:
        logging.error(f"Error in training Random Forest model: {e}. Please ensure that the training data is properly formatted and parameters are correctly specified.")
        raise

<<<<<<< HEAD
def evaluate_model(rf_model, X_test, y_test):
    """
    Evaluate the trained Random Forest model
    """
        
    try:
        # Out-of-bag (OOB) score
        oob_score = rf_model.oob_score_
        logging.info(f"OOB Score: {oob_score}")
=======
# Define and configure the RandomForestRegressor with regularization
rf_model = RandomForestRegressor(
    n_estimators=100,  # Keep the number of trees to 100
    max_depth=10,  # Limit the depth of each tree to prevent overfitting
    min_samples_split=5,  # Require at least 5 samples to split a node
    min_samples_leaf=2,  # Ensure at least 2 samples exist in each leaf node
    max_features='sqrt',  # Limit the number of features considered at each split
    random_state=42,
    oob_score=True #Set to true to collect model's accuracy on unseen data (OOB samples)
)
>>>>>>> 2bfa1e1eccde5f892f41448175222fad7922d6e0

        # Mean Squared Error
        y_pred = rf_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        logging.info(f'Mean Squared Error with Regularization: {mse}')

<<<<<<< HEAD
        # Mean Absolute Error
        mae = mean_absolute_error(y_test, y_pred)
        logging.info(f'Mean Absolute Error with Regularization: {mae}')
=======
# Calculate and print the OOB score 
oob_score = rf_model.oob_score_
print(f"OOB Score: {oob_score}")

# Make predictions on the test set
y_pred = rf_model.predict(X_test)
>>>>>>> 2bfa1e1eccde5f892f41448175222fad7922d6e0

        # Confusion Matrix
        y_test_pred = (y_pred > 0.5).astype(int)
        cm = confusion_matrix(y_test, y_test_pred)
        logging.info(f"Confusion Matrix:\n{cm}")
    except Exception as e:
        logging.error(f"Error in evaluating the model: {e}. Please ensure the test data is properly formatted and the evaluation metrics are appropriately defined.")
        raise

<<<<<<< HEAD
def plot_feature_importance(rf_model, X_train):
    """
    Plot feature importances of the  Random Forest model
    """
    try:
        # Feature Importance
        feature_importances = rf_model.feature_importances_
        feature_names = X_train.columns
        sorted_indices = np.argsort(feature_importances)[::-1]  # sort in descending order
        sorted_features = [(feature_names[i], feature_importances[i]) for i in sorted_indices]
=======
# Calculate the confusion matrix
y_test_pred = (y_pred > 0.5).astype(int)
cm = confusion_matrix(y_test, y_test_pred)
print("Confusion Matrix:\n", cm)

# Get feature importance from the trained model
feature_importances = rf_model.feature_importances_
feature_names = X_train.columns
>>>>>>> 2bfa1e1eccde5f892f41448175222fad7922d6e0

        logging.info("\nFeature Importances (with Regularization):")
        for feature, importance in sorted_features:
            logging.info(f"{feature}: {importance:.4f}")

        # Plot Feature Importance
        plt.figure(figsize=(12, 6))
        sns.barplot(x=[feature[1] for feature in sorted_features], y=[feature[0] for feature in sorted_features])
        plt.xlabel("Feature Importance Score")
        plt.ylabel("Features")
        plt.title("Feature Importance - Random Forest (With Regularization)")
        plt.show()
    except Exception as e:
        logging.error(f"Error in plotting feature importance: {e}. Please verify the feature names,importance scores and ensure the plotting libraries are properly imported.")
        raise

def main():
    # Load dataset
    data = load_dataset('cleaned_data.csv')

    # Preprocess data
    date_columns = ['date_applied', 'date_sourced']
    drop_column = ['date_rejected']
    data = preprocess_data(data, date_columns, drop_column)

    # Separate target variable before encoding
    y = data.pop('rejected')

    # Identify categorical columns
    categorical_cols = ['company_name', 'position', 'job_description', 'location', 'technical_test', 'interview', 'applied',
                        'contract_type', 'language', 'job_location', 'mode_of_application', 'job_delisted']

    # Encode features
    x_encoded = encode_features(data, y, categorical_cols)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(x_encoded, y, test_size=0.2, random_state=42)

    # Configure and train Random Forest model
    rf_model = train_random_forest(X_train, y_train, 
        n_estimators=100, 
        max_depth=10, 
        min_samples_split=5, 
        min_samples_leaf=2, 
        max_features='sqrt')

    # Evaluate model performance
    evaluate_model(rf_model, X_test, y_test)

    # Plot Feature Importance
    plot_feature_importance(rf_model, X_train)

if __name__ == "__main__":
    main()