import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
from category_encoders import TargetEncoder

# Define log information format
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

def preprocess_data(data, date_columns, drop_column):
    """
    Preprocess the dataset
    """
    try:
        # Remove unnecessary column
        data.drop(columns=drop_column, inplace=True)
        logging.info(f"Removed column: {drop_column}.")

        # Convert date strings to datetime objects and extract month
        for col in date_columns:
            data[col] = pd.to_datetime(data[col], format='%Y-%m-%d', errors='coerce')
            data[f'{col}_month'] = data[col].dt.month.fillna(0).astype(int)
            logging.info(f"Processed column {col}.")

        data.drop(columns=date_columns, inplace=True)
        logging.info(f"Dropped original date columns: {date_columns}.")

        # Replace missing values in 'date_applied_month' using the average of 'date_sourced_month'
        month_avg = data.groupby('date_sourced_month')['date_applied_month'].mean()
        data['date_applied_month'] = data.apply(
            lambda row: month_avg.get(row['date_sourced_month'], 0) if pd.isna(row['date_applied_month']) else row['date_applied_month'],
            axis=1
        )
        logging.info("Filled missing values in 'date_applied_month'.")
        
        return data
    except KeyError as e:
        logging.error(f"KeyError in preprocessing data: {e}. Ensure the column names are correct.")
        raise
    except Exception as e:
        logging.error(f"Unexpected error in preprocessing data: {e}.")
        raise

def encode_features(data, target, categorical_cols):
    """
    Apply target encoding to categorical features.
    """
    try:
        encoder = TargetEncoder(cols=categorical_cols)
        x_encoded = encoder.fit_transform(data, target)
        logging.info(f"Applied target encoding to columns: {categorical_cols}.")
        return x_encoded
    except Exception as e:
        logging.error(f"Error in encoding features with TargetEncoder: {e}.")
        raise

def train_random_forest(X_train, y_train, **kwargs):
    """
    Configure and train the Random Forest model.
    """
    try:
        rf_model = RandomForestClassifier(random_state=42, **kwargs)
        rf_model.fit(X_train, y_train)
        logging.info("Trained Random Forest model.")
        return rf_model
    except Exception as e:
        logging.error(f"Error in training Random Forest model: {e}.")
        raise

def evaluate_model(rf_model, X_test, y_test):
    """
    Evaluate the trained Random Forest model.
    """
    try:
        # Out-of-bag (OOB) score
        oob_score = rf_model.oob_score_
        logging.info(f"OOB Score: {oob_score}")       
        
        # Get predictions
        y_pred = rf_model.predict(X_test)
        y_pred_proba = rf_model.predict_proba(X_test)[:, 1]  # For ROC-AUC score

        # Confusion Matrix
        y_test_pred = (y_pred > 0.5).astype(int)
        cm = confusion_matrix(y_test, y_test_pred)
        logging.info(f"Confusion Matrix:\n{cm}")

        # Precision
        precision = precision_score(y_test, y_pred)
        logging.info(f"Precision: {precision}")

        # Recall
        recall = recall_score(y_test, y_pred)
        logging.info(f"Recall: {recall}")

        # F1-Score
        f1 = f1_score(y_test, y_pred)
        logging.info(f"F1-Score: {f1}")

        # ROC-AUC Score
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        logging.info(f"ROC-AUC Score: {roc_auc}")

    except Exception as e:
        logging.error(f"Error in evaluating the model: {e}")
        raise

def plot_feature_importance(rf_model, X_train):
    """
    Plot feature importances of the Random Forest model.
    """
    try:
        feature_importances = rf_model.feature_importances_
        feature_names = X_train.columns
        sorted_indices = np.argsort(feature_importances)[::-1]
        sorted_features = [(feature_names[i], feature_importances[i]) for i in sorted_indices]

        logging.info("Feature Importances:")
        for feature, importance in sorted_features:
            logging.info(f"{feature}: {importance:.4f}")

        # Plot feature importance
        plt.figure(figsize=(12, 6))
        sns.barplot(x=[feature[1] for feature in sorted_features], y=[feature[0] for feature in sorted_features])
        plt.xlabel("Feature Importance Score")
        plt.ylabel("Features")
        plt.title("Feature Importance - Random Forest")
        plt.show()
    except Exception as e:
        logging.error(f"Error in plotting feature importance: {e}.")
        raise

def main():
    try:
        # Load dataset
        data = load_dataset('cleaned_data.csv')

        # Preprocess data
        date_columns = ['date_applied', 'date_sourced']
        drop_column = 'date_rejected'
        data = preprocess_data(data, date_columns, drop_column)

        # Separate target variable
        y = data.pop('rejected')

        # Identify categorical columns
        categorical_cols = ['company_name', 'position', 'job_description', 'location', 'technical_test', 'interview', 'applied',
                            'contract_type', 'language', 'job_location', 'mode_of_application', 'job_delisted']

        # Encode features
        X_encoded = encode_features(data, y, categorical_cols)

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

        # Train Random Forest model
        rf_model = train_random_forest(X_train, y_train, 
                                       n_estimators=100, 
                                       max_depth=10, 
                                       min_samples_split=5, 
                                       min_samples_leaf=2, 
                                       max_features='sqrt',
                                       oob_score=True)

        # Evaluate the model
        evaluate_model(rf_model, X_test, y_test)

        # Plot feature importance
        plot_feature_importance(rf_model, X_train)

    except Exception as e:
        logging.error(f"An error occurred in the main function: {e}.")
        raise

if __name__ == "__main__":
    main()
