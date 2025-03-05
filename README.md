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

1. Regularization:

- Experiment with different regularization parameters to find the optimal balance.

2. Cross-Validation:

- Use k-fold cross-validation to evaluate the model's performance on different subsets of the data.

3. Simplify the Model:

- Reduce the complexity of the model by decreasing the number of estimators or depth of trees in the Random Forest Regressor.

- Experiment with different hyperparameters to find a simpler model that performs well.

  Changes Made (February 5, 2025)

These approaches may all be used or not, depending on what would best generate the best model. As this is a work in 
progress and would use several methods that will alter many parts of the original code, each changes to the code would be uploaded in seprate files. 

![Button1](https://img.shields.io/badge/UPDATES-Regularization%20of%20the%20Model-red)

```
# Define and configure the RandomForestRegressor with regularization
rf_model = RandomForestRegressor(
    n_estimators=100,  # Keep the number of trees to 100
    max_depth=10,  # Limit the depth of each tree to prevent overfitting
    min_samples_split=5,  # Require at least 5 samples to split a node
    min_samples_leaf=2,  # Ensure at least 2 samples exist in each leaf node
    max_features='sqrt',  # Limit the number of features considered at each split
    random_state=42
```
- By using max-depth of 10, the decisions trees don't grow too complex and memorize the training data to avoid overfitting.
- By requiring at least 5 samples to split the node, this avoids splits that result in overly small leaf nodes, improving generalization.
- By using min_samples_split of 2, this ensures that the tree doesn't get too specific to the training data.
- By using max_features= 'sqrt', it introduces randomness and helps reduce overfitting by ensuring trees don’t rely too heavily on any particular feature

With this regularization, the model learns from more variables compared to the first training where it focused on one categorical variable. Here's the Feature Importance with the regularization of the RF model. 

![Image2](https://github.com/rnx2024/Predict-Job-Rejection-Using-Random-Forest/blob/main/Model-Enhancements/feature-importance-regularization.png)

