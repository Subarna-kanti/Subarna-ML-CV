#importing necessary packages
import pandas as pd
pd.set_option('display.max_columns', 21)#Changing default display option to display all columns
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from scipy import stats#Stats package for statistical analysis
#Machine learning packages
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.utils import resample
from sklearn.metrics import roc_curve, roc_auc_score, auc, accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split 

df=pd.read_csv('bank_customer_survey.csv')
df.shape

#Dropping the duplicates
df = df.drop_duplicates()

#Dataframe dimensions
df.shape
#no duplicate values

df.info()
#no null values

#Selecting categorical columns
categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
#Looping through the columns and changing type to 'category'
for column in categorical_columns:
    df[column] = df[column].astype('category')

df.info()
df.head(15)
df.tail(15)

#Bar plots of categorical features
for feature in df.dtypes[df.dtypes == 'category'].index:
    sns.countplot(y=feature, data=df, order = df[feature].value_counts().index)
    plt.show()

#job status as proportion of overall number of values
df.job.value_counts()/45211

#default credit status as proportion of overall number of values
df.default.value_counts()/45211

#housing loan status as proportion of overall number of values
df.housing.value_counts()/45211

#personal status as proportion of overall number of values
df.loan.value_counts()/45211

#maritial status as proportion of overall number of values
df.marital.value_counts()/45211

#education status as proportion of overall number of values
df.education.value_counts()/45211

#month status as proportion of overall number of values
df.month.value_counts()/45211

#previous outcome status as proportion of overall number of values
df.poutcome.value_counts()/45211

#Histogram grid
df.hist(figsize=(10,10), xrot=-45)

#Clear the text "residue"
plt.show()

#summary of numeric features
df.describe()

#Creating a copy of the original data frame
df_cleaned = df.copy()

#Dropping the unknown job level
df_cleaned = df_cleaned[df_cleaned.job != 'unknown']

#Dropping the unknown marital status
df_cleaned = df_cleaned[df_cleaned.marital != 'unknown']

#Dropping the unknown and illiterate education level
df_cleaned = df_cleaned[df_cleaned.education != 'unknown']

#Deleting the 'default' column
del df_cleaned['default']

#Deleting the 'duration' column
del df_cleaned['duration']

df_cleaned.head(15)
df_cleaned.shape

a=df_cleaned.poutcome.value_counts()
b=df_cleaned.pdays.value_counts()
print(a['unknown'],b[-1])

c=df_cleaned.previous.value_counts()
print(c[0])

no_match=df_cleaned.loc[((df_cleaned['pdays']!=-1)&(df_cleaned['poutcome']=='unknown'))]
no_match.pdays.value_counts()

#Getting the positions of the mistakenly labeled 'pdays'
x = df_cleaned.loc[(df_cleaned['pdays'] !=-1) & (df['poutcome'] == 'unknown')]['pdays'].index.values

#Assigning NaNs instead of '-1'
df_cleaned.loc[x,'pdays'] = np.nan

#Dropping NAs from the dataset
df_cleaned = df_cleaned.dropna()

a=df_cleaned.poutcome.value_counts()
b=df_cleaned.pdays.value_counts()
print(a['unknown'],b[-1])

#Saving the cleaned dataset as a file
df_cleaned.to_csv('cleaned_data.csv')
df_clean=pd.read_csv('cleaned_data.csv')
df_clean.head()
del df_clean['Unnamed: 0']

#Calculate correlations between numeric features
correlations = df_clean.corr()
#Make the figsize 7 x 6
plt.figure(figsize=(7,6))
_ = sns.heatmap(correlations, cmap="Reds")#heatmap

df_clean1 = pd.get_dummies(df_clean, drop_first=True)
df_clean1.head()

#Splitting variables into predictor and target variables
X=df_clean1.drop('y', axis=1)
y=df_clean1.y

#Setting up pipelines with a StandardScaler function to normalize variables
pipelines = {
    'log1':make_pipeline(StandardScaler(),LogisticRegression(penalty='l1',random_state=42,class_weight='balanced')),
    'log2' : make_pipeline(StandardScaler(), LogisticRegression(penalty='l2',random_state=42,class_weight='balanced')),
    #Setting the penalty for simple Logistic Regression as L2 to minimize the fitting time
    'log_reg' : make_pipeline(StandardScaler(), LogisticRegression(penalty='l2', random_state=42, class_weight='balanced'))
}

#Setting up a very large hyperparameter C for the non-penalized Logistic Regression (to cancel the regularization)
log_reg_hyperparameters={
    'logisticregression__C':np.linspace(100000, 100001, 1),
    'logisticregression__fit_intercept':[True, False]
}

#Setting up hyperparameters for the Logistic Regression with log1 penalty
log1_hyperparameters = {
    'logisticregression__C' : np.linspace(1e-3, 1e3, 10),
    'logisticregression__fit_intercept' : [True, False]
}

#Setting up hyperparameters for the Logistic Regression with log2 penalty
log2_hyperparameters={
    'logisticregression__C':np.linspace(1e-3, 1e3, 10),
    'logisticregression__fit_intercept':[True, False]
}

#Creating the dictionary of hyperparameters
hyperparameters = {
    'log_reg' : log_reg_hyperparameters,
    'log1' : log1_hyperparameters,
    'log2' : log2_hyperparameters
}

#Splitting the data into train and test sets
X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.3, random_state=42)

#Creating an empty dictionary for fitted models
fitted_logreg_models={}

# Looping through model pipelines, tuning each with GridSearchCV and saving it to fitted_logreg_models
for name, pipeline in pipelines.items():
    #Creating cross-validation object from pipeline and hyperparameters
    model = GridSearchCV(pipeline, hyperparameters[name],cv=10,n_jobs=-1)
    #Fitting the model on X_train, y_train
    model.fit(X_train, y_train)
    #Storing the model in fitted_logreg_models[name] 
    fitted_logreg_models[name] = model
    #Printing the status of the fitting
    print(name, 'fitted.')

#Displaying best score for each fitted model
for name,model in fitted_logreg_models.items():
    print(name,model.best_score_)

#Creating an empty dictionary for predicted models
predicted_logreg_models = {}

#Predicting the response variables and displaying the prediction score
for name, model in fitted_logreg_models.items():
    y_pred = model.predict(X_test)
    predicted_logreg_models[name] = accuracy_score(y_test, y_pred)
print(predicted_logreg_models)

#defining the model with the highest accuracy score
max(predicted_logreg_models,key=lambda k:predicted_logreg_models[k])

#Creating the confusion matrix
pd.crosstab(y_test,fitted_logreg_models['log1'].predict(X_test),rownames=['True'],colnames=['Predict'],margins=True)

#Creating the classification report
print(classification_report(y_test, fitted_logreg_models['log1'].predict(X_test)))

#Obtaining the ROC score
roc_auc = roc_auc_score(y_test, fitted_logreg_models['log1'].predict(X_test))

#Obtaining false and true positives & thresholds
fpr, tpr, thresholds = roc_curve(y_test, fitted_logreg_models['log1'].predict_proba(X_test)[:,1])

#Plotting the curve
plt.plot(fpr, tpr, label='log 1 Logistic Regression (area = %0.03f)' % roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC curve for Logistic regression')
plt.legend(loc="upper left")
plt.show()


#Setting up pipelines with a StandardScaler function to normalize the variables
pipelines = {
    'rf' : make_pipeline(StandardScaler(),RandomForestClassifier(random_state=42, class_weight='balanced')),
    'gb' : make_pipeline(StandardScaler(),GradientBoostingClassifier(random_state=42))
}

#Setting up hyperparameters for the Random Forest
rf_hyperparameters = {
    'randomforestclassifier__n_estimators': [100, 200],
    'randomforestclassifier__max_features': ['auto', 'sqrt', 0.33]
}

#Setting up hyperparameters for the Gradient Boost
gb_hyperparameters = {
    'gradientboostingclassifier__n_estimators': [100, 200],
    'gradientboostingclassifier__learning_rate': [0.05, 0.1, 0.2],
    'gradientboostingclassifier__max_depth': [1, 3, 5]
}

#Creating the dictionary of hyperparameters
hyperparameters = {
    'rf' : rf_hyperparameters,
    'gb' : gb_hyperparameters
}

#Creating an empty dictionary for fitted models
fitted_alternative_models = {}

# Looping through model pipelines, tuning each with GridSearchCV and saving it to fitted_logreg_models
for name, pipeline in pipelines.items():
    #Creating cross-validation object from pipeline and hyperparameters
    alt_model = GridSearchCV(pipeline, hyperparameters[name], cv=10, n_jobs=-1)
    
    #Fitting the model on X_train, y_train
    alt_model.fit(X_train, y_train)
    
    #Storing the model in fitted_logreg_models[name] 
    fitted_alternative_models[name] = alt_model
    
    #Printing the status of the fitting
    print(name, 'fitted.')

#Displaying the best_score_ for each fitted model
for name, model in fitted_alternative_models.items():
    print(name, model.best_score_ )

#Creating the confusion matrix for Random Forest
pd.crosstab(y_test, fitted_alternative_models['rf'].predict(X_test), rownames=['True'], colnames=['Predicted'], margins=True)

#Creating the classification report for Gradient Boosting
print(classification_report(y_test, fitted_alternative_models['gb'].predict(X_test)))
# Looping through model pipelines, tuning each with GridSearchCV and saving it to fitted_logreg_models
for name, pipeline in pipelines.items():
    #Creating cross-validation object from pipeline and hyperparameters
    alt_model = GridSearchCV(pipeline, hyperparameters[name], cv=10, n_jobs=-1)
    
    #Fitting the model on X_test, y_test
    alt_model.fit(X_train, y_train)
    
    #Storing the model in fitted_logreg_models[name] 
    fitted_alternative_models[name] = alt_model
    
    #Printing the status of the fitting
    print(name, 'fitted.')

#Displaying the best_score_ for each fitted model
for name, model in fitted_alternative_models.items():
    print(name, model.best_score_ )

#Creating the confusion matrix for Random Forest
pd.crosstab(y_test, fitted_alternative_models['rf'].predict(X_test), rownames=['True'], colnames=['Predicted'], margins=True)

#Creating the classification report for Gradient Boosting
print(classification_report(y_test, fitted_alternative_models['gb'].predict(X_test)))

