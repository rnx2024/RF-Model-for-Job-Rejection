## Predicting Job Rejections using Random Forest Model

This project trains an RF model to predict rejections using a dataset of more than 25K jobs. The data is preprocessed and fed into 
a Random Forest Regressor model to predict the probability of rejection based on various features. 

This is to make clear that in training the model no applicant data is used. Without considering applicant-specific details like 
qualifications, skills, or experience, predictions are far less accurate. It would reduce the process to broad generalizations 
that don't account for individual merit. However, it is still useful in the context of how machine learing model learns from data. That
while the model may not have used applicant data, it has the ability to learn the primary factors in job rejections. As such, this project 
merely wants to illustrate the process of training RF model with raw data and how to avoid overfitting.

# Features

1. Data Preprocessing:

  - Converts date columns to datetime objects and extracts the month.

  - Applies target encoding to categorical features.

  - Handles missing values and ensures all features are numeric.

2. Model Training:

  - Uses Random Forest Regressor for prediction.

  - Evaluates the model using Mean Squared Error (MSE), OOB (Out of Bag) Score and Confusion Matrix

  - Displays feature importances.

## Instructions on how to Use Repository

1. Clone the repository 

``` git clone https://github.com/rnx2024/Job-Rejection-Prediction```

2. Install dependencies using requirements.txt
   
```pip install -r requirements.txt```

4. Run the Python [file](https://github.com/rnx2024/Job-Rejection-Prediction/blob/main/RF_Predict_Job_Rejections.py)

As this is the first training for the Random Forest Model, it shows signs of overfitting. 
The graph below shows how the RF model played too much focus on one categorical variable: 

![Image](https://github.com/rnx2024/Job-Rejection-Prediction/blob/main/feature_importance_firstprediction.png)

To address the overfitting, the project configured the RF model and implemented regularization.

This initial update provides a code [file](https://github.com/rnx2024/Predict-Job-Rejection-Using-Random-Forest/blob/main/Model-Enhancements/RF_model_regularized.py) that regularized the model.

```
# Define and configure the RandomForestRegressor with regularization
rf_model = RandomForestRegressor(
    n_estimators=100,  # Keep the number of trees to 100
    max_depth=10,  # Limit the depth of each tree to prevent overfitting
    min_samples_split=5,  # Require at least 5 samples to split a node
    min_samples_leaf=2,  # Ensure at least 2 samples exist in each leaf node
    max_features='sqrt',  # Limit the number of features considered at each split
    random_state=42,
    oob_score=True
)
```
- By using max-depth of 10, the decisions trees don't grow too complex and memorize the training data to avoid overfitting.
- By requiring at least 5 samples to split the node, this avoids splits that result in overly small leaf nodes, improving generalization.
- By using min_samples_split of 2, this ensures that the tree doesn't get too specific to the training data.
- By using max_features= 'sqrt', it introduces randomness and helps reduce overfitting by ensuring trees don’t rely too heavily on any particular feature
- OOB_Score is set to True to use the data for assessing the model's accuracy on the unseen data or OOB samples

```
oob_score = rf_model.oob_score_
print(f"OOB Score: {oob_score}")
```
A calculation for the confusion matrix to see a detailed performance of the model with regards to True Negatives and False Positives.

```
# Calculate the confusion matrix
y_test_pred = (y_pred > 0.5).astype(int)
cm = confusion_matrix(y_test, y_test_pred)
print("Confusion Matrix:\n", cm)
```
The confusion matrix indicated: 

```
Confusion Matrix:
 [[3095   39] # True Negatives (39 predicted rejections but actual non-rejections)
 [  11 1911]] # False Positives (11 predicted non-rejections but actual rejections)
 ```
This means that the model has reliable predictions with minimal errors.

THe fine-tuned RF model has an
**OOB Score: 0.9635058515086417** and a **Mean Squared Error: 0.007783707629690703** which means that it is performing well on unseen data.

With this regularization, the model learns from more variables compared to the first training. Here's the Feature Importance with the regularization of the RF model. 

![Image2]([https://github.com/rnx2024/Predict-Job-Rejection-Using-Random-Forest/blob/main/Model-Enhancements/feature-importance-regularization.png])

