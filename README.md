## Predicting Job Rejections using Random Forest Model

The data is preprocessed and fed into a Random Forest Regressor model to predict the probability of rejection based on various features.

# Features

1. Data Preprocessing:

  - Converts date columns to datetime objects and extracts the month.

  - Applies target encoding to categorical features.

  - Handles missing values and ensures all features are numeric.

2. Model Training:

  - Uses Random Forest Regressor for prediction.

  - Evaluates the model using Mean Squared Error (MSE).

  - Displays feature importances.

## Instructions on how to Use Repository

1. Clone the repository and install dependencies using requirements.txt

``` git clone https://github.com/rnx2024/Job-Rejection-Prediction```
```pip install -r requirements.txt```

2. Run the Python file Predict-Job-Rejections.py

