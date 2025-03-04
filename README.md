# Diabetes Prediction Model

This repository contains a machine learning model to predict the likelihood of diabetes based on various patient health features. The dataset used is `diabetes.csv`, which contains patient information like glucose levels, BMI, age, and other health-related factors.

## Project Overview

This project uses different machine learning algorithms to classify patients as either having diabetes or not. Various models are trained, including logistic regression, decision trees, random forests, support vector machines, k-nearest neighbors, gradient boosting, Naive Bayes, and neural networks.

The key objectives of the project are:
1. Data preprocessing and exploratory data analysis (EDA).
2. Model training and evaluation using different classification algorithms.
3. Identifying the best model based on accuracy and other evaluation metrics.

## Dataset

The `diabetes.csv` dataset includes the following columns:

1. **Pregnancies**: Number of pregnancies the patient has had.
2. **Glucose**: Plasma glucose concentration after a 2-hour oral glucose tolerance test.
3. **BloodPressure**: Diastolic blood pressure (mm Hg).
4. **SkinThickness**: Triceps skin fold thickness (mm).
5. **Insulin**: 2-hour serum insulin (mu U/ml).
6. **BMI**: Body mass index (weight in kg / (height in m)^2).
7. **DiabetesPedigreeFunction**: Diabetes pedigree function (a measure of diabetes risk based on family history).
8. **Age**: Age of the patient in years.
9. **Outcome**: Class variable (0 = no diabetes, 1 = diabetes).

## Installation

To use this repository, clone it and install the necessary dependencies using the following:

```bash
git clone https://github.com/Gauthamraj-S/Diabetes.git
cd Diabetes
pip install -r requirements.txt
```

### Dependencies

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## How to Use

1. **Load and Explore Data**: 
   The script loads the diabetes dataset and performs initial exploratory analysis (checking for null values, duplicates, and statistical summaries). 
   
   ```python
   df = pd.read_csv('diabetes.csv')
   df.head()  # Preview the data
   ```

2. **Data Preprocessing**:
   The data is split into numerical and categorical features, and the features are scaled using `StandardScaler`.

   ```python
   X = df.drop(columns=['Outcome'])  # Feature matrix
   y = df['Outcome']  # Target variable
   ```

3. **Model Training and Evaluation**:
   Multiple machine learning models are trained, and their performance is evaluated using accuracy, confusion matrix, and classification reports.

   ```python
   # Train a Logistic Regression model
   lr = LogisticRegression(max_iter=1000)
   lr.fit(X_train, y_train)
   prediction = lr.predict(X_test)
   ```

   This process is repeated for different classifiers:
   - Decision Tree
   - Random Forest
   - Support Vector Machine (SVM)
   - K-Nearest Neighbors (KNN)
   - Gradient Boosting
   - Naive Bayes
   - Neural Networks (MLP)

4. **Results**:
   After training the models, the script prints the accuracy and evaluation metrics for each classifier. The final accuracy is reported, and the best performing model is identified.

   ```python
   print(f"Accuracy: {accuracy_score(y_test, prediction)}")
   ```

5. **Visualizations**:
   Various plots are generated to better understand the data, including histograms, box plots, heatmaps, pair plots, and more.

   ```python
   sns.pairplot(df, hue='Outcome', vars=numerical_features, palette='coolwarm')
   ```

## Conclusion

The **Gaussian Naive Bayes** classifier achieved the highest accuracy of **76.62%** for this dataset. The various models showed different performances, and further tuning could improve results.


Feel free to explore, modify, and experiment with the code.