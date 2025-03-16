## Predicting Job Rejections using Random Forest Model

This project fine-tune an RF model to predict rejections using a dataset of more than 25K jobs. The data is preprocessed and fed into 
A Random Forest model to predict the probability of rejection based on various features. 

This is to make clear that in training the model no applicant data is used. Without considering applicant-specific details like 
qualifications, skills, or experience, predictions would reduce the process to broad generalizations that don't account for individual 
merit. However, this project is useful in the context of how machine learning model learns from data. That while the model may not have 
used applicant data, it can learn from data and make predictions with high accuracy. 

### Main Features 

**1. Data Preprocessing:**

  - Converts date columns to datetime objects and extracts the month.

  - Applies target encoding to categorical features.

  - Handles missing values and ensures all features are numeric.


**2. Implemented regularization for higher accuracy**

```
# Define and configure the RandomForestClassifier with regularization
rf_model = RandomForestRegressor(
    n_estimators=100,  # Keep the number of trees to 100
    max_depth=10,  # Limit the depth of each tree to prevent overfitting
    min_samples_split=5,  # Require at least 5 samples to split a node
    min_samples_leaf=2,  # Ensure at least 2 samples exist in each leaf node
    max_features='sqrt',  # Limit the number of features considered at each split
    random_state=42
    oob_score=True,
)
```
- By using max-depth of 10, the decisions trees don't grow too complex and memorize the training data to avoid overfitting.
- By requiring at least 5 samples to split the node, this avoids splits that result in overly small leaf nodes, improving generalization.
- By using min_samples_split of 2, this ensures that the tree doesn't get too specific to the training data.
- By using max_features= 'sqrt', it introduces randomness and helps reduce overfitting by ensuring trees don’t rely too heavily on any particular feature

**3. Evaluation Metrics**

- **OOB Score:** 0.9635058515086417 (This evaluates the RF model's ability to predict unseen data during training by using the Out-of-Bag samples)

- **Confusion Matrix:** 
[[3095   39] # True Negatives (39 predicted rejections but actual non-rejections)
 [  11 1911]] # False Positives (11 predicted non-rejections but actual rejections)

- **Recall:** 0.9942767950052029 (This is the model's ability to identify actual rejections) 
- **Precision:** 0.98 (How well it avoids false positives)
- **F1-score:** 0.9870867768595041 (This is to check the balance between the false negatives and false positives)
- **ROC-AUC score**: 0.9994095672517259 (Guage the model's efficiency in differentiating rejected and not rejected)

With the evalution results, the model shows that it has reliable predictions with high accuracy and minimal errors. 

**4. Plotted Feature Importance**

![Image1](https://github.com/rnx2024/RF-Model-for-Job-Rejection/blob/main/feature-importance.png)


## Usage Instructions

1. Clone the repository 

``` git clone https://github.com/rnx2024/RF-Model-for-Job-Rejection```

2. Install dependencies using requirements.txt
   
```pip install -r requirements.txt```

4. Run the Python [file](https://github.com/rnx2024/RF-Model-for-Job-Rejection/blob/main/RF_model_regularized.py)

