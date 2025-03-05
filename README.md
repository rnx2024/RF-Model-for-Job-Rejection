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

1. Clone the repository 

``` git clone https://github.com/rnx2024/Job-Rejection-Prediction```

2. Install dependencies using requirements.txt
   
```pip install -r requirements.txt```

4. Run the Python [file](https://github.com/rnx2024/Job-Rejection-Prediction/blob/main/RF_Predict_Job_Rejections.py)

As this is the first training for the Random Forest Model, it shows signs of overfitting. The graph below shows how the RF model played too much focus on one categorical variable: 
![Image](https://github.com/rnx2024/Job-Rejection-Prediction/blob/main/feature_importance_firstprediction.png)

To address the overfitting, the project intends to do the following approaches: 


1. Increase Data Volume:

- Collect more training data to help the model learn a broader range of patterns.

2. Feature Selection:

- Remove irrelevant or less important features that do not contribute significantly to the model's predictions.

3. Regularization:

- Experiment with different regularization parameters to find the optimal balance.

4. Cross-Validation:

- Use k-fold cross-validation to evaluate the model's performance on different subsets of the data.

5. Simplify the Model:

- Reduce the complexity of the model by decreasing the number of estimators or depth of trees in the Random Forest Regressor.

- Experiment with different hyperparameters to find a simpler model that performs well.

6. Early Stopping:

- Monitor the model's performance on a validation set during training and stop training when the performance starts to degrade.

These approaches may all be used or not, depending on what would best generate the best model. As this is a work in progress and would use several methods that will alter many parts of the original code, each changes to the code would be uploaded in seprate files. 
