# Data Science The Estonia Disaster Passenger List: Project Overview 
* Handling imbalanced Datasets with Oversample techniques (SMOTE)
* Cross-validation using StratifiedKFold
* Optimized Logistic Regression, KNears, SVC, Decision tree, Random Forest and XGBC using GridsearchCV to reach the best model. 


## Code and Resources Used 
**Python Version:** 3.8.2
**Packages:** pandas, numpy, sklearn, matplotlib, seaborn, plotly, imblearn, warnings, xgboost  
**Inspirational Github:** https://github.com/PlayingNumbers/ds_salary_proj/  
**SMOTE with cross-validation Notebook:** https://www.kaggle.com/janiobachmann/credit-fraud-dealing-with-imbalanced-datasets  
**Data Set:** https://www.kaggle.com/christianlillelund/passenger-list-for-the-estonia-ferry-disaster  


## EDA
I looked at the distributions of the data and the value counts for the various categorical variables. Below are a few highlights from the charts. 

![alt text](https://github.com/RodolfoSales/Projeto1/blob/master/figures.jpeg "Figueres")
![alt text](https://github.com/RodolfoSales/Projeto1/blob/master/sunburst.png "SunBurst")
![alt text](https://github.com/RodolfoSales/Projeto1/blob/master/correlationmatrix.jpeg "Correlation matrix")

## Model Building 

First, I transformed the categorical variables using LabelEncoder in order to work with it. I also split the data into train and tests sets using Stratified K-Folds cross-validator. After I applied the SMOTE technique during cross-validation to minimize Overfitting.  

I tried six different models and evaluated them using Confusion Matrix and AUC score. I chose to use Confusion Matrix to see the relationship among TP, TN, FP and FN.   

## Model performance
The Logistic Regression model outperformed the other approaches on the test and validation sets. 
*	**Random Forest** : AUC = 0.61
*	**Logistic Regression**: AUC = 0.77
*	**KNearest**: AUC = 0.61
*	**SVC**: AUC = 0.74
*	**Decision Tree**: AUC = 0.71
*	**XGBC**: AUC = 0.65


