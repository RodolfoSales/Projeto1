# Data Science The Estonia Disaster Passenger List: Project Overview 
* Handling imbalanced Datasets with Oversample techniques (SMOTE)
* Cross-validation using StratifiedKFold
* Optimized Logistic Regression, KNears, SVC, Decision tree, Random Forest and XGBC Regressors using GridsearchCV to reach the best model. 


## Code and Resources Used 
**Python Version:** 3.8.2
**Packages:** pandas, numpy, sklearn, matplotlib, seaborn, plotly, imblearn, warnings, xgboost  
**Inspirational Github:** https://github.com/PlayingNumbers/ds_salary_proj/  
**SMOTE with cross-validation Notebook:** https://www.kaggle.com/janiobachmann/credit-fraud-dealing-with-imbalanced-datasets  
**Data Set:** https://www.kaggle.com/christianlillelund/passenger-list-for-the-estonia-ferry-disaster  


## EDA
I looked at the distributions of the data and the value counts for the various categorical variables. Below are a few highlights from the charts. 

![alt text](https://github.com/PlayingNumbers/ds_salary_proj/blob/master/salary_by_job_title.PNG "Salary by Position")
![alt text](https://github.com/PlayingNumbers/ds_salary_proj/blob/master/positions_by_state.png "Job Opportunities by State")
![alt text](https://github.com/PlayingNumbers/ds_salary_proj/blob/master/correlation_visual.png "Correlations")

## Model Building 

First, I transformed the categorical variables into dummy variables. I also split the data into train and tests sets with a test size of 20%.   

I tried three different models and evaluated them using Mean Absolute Error. I chose MAE because it is relatively easy to interpret and outliers aren’t particularly bad in for this type of model.   

I tried three different models:
*	**Multiple Linear Regression** – Baseline for the model
*	**Lasso Regression** – Because of the sparse data from the many categorical variables, I thought a normalized regression like lasso would be effective.
*	**Random Forest** – Again, with the sparsity associated with the data, I thought that this would be a good fit. 

## Model performance
The Random Forest model far outperformed the other approaches on the test and validation sets. 
*	**Random Forest** : AUC = 11.22
*	**Linear Regression**: AUC = 18.86
*	**Ridge Regression**: AUC = 19.67


