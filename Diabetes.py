#Importing necessary packages
import pandas as pd
import numpy as np
import seaborn as sns
sns.set(color_codes =True)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

#Importing the Diabetes CSV data file
data=pd.read_csv('https://raw.githubusercontent.com/BukkaNikhilSai/ML_TA_IIITB_2019/master/Assignment_1/Pima_Indian_diabetes.csv')
data

data.info()

data.describe()

#histogram of column Pregnancies
fig=plt.figure(figsize=(17,10))
data.hist(column="Pregnancies")
plt.xlabel("Pregnancies",fontsize=15)
plt.ylabel("Frequency",fontsize=15)
plt.show()
#histogram of column glucose
fig=plt.figure(figsize=(17,10))
data.hist(column="Glucose")
plt.xlabel("Glucose",fontsize=15)
plt.ylabel("Frequency",fontsize=15)
plt.show()
#histogram of column Blood Pressure
fig=plt.figure(figsize=(17,10))
data.hist(column="BloodPressure")
plt.xlabel("BloodPressure",fontsize=15)
plt.ylabel("Frequency",fontsize=15)
plt.show()
#histogram of Skin thickness
fig=plt.figure(figsize=(17,10))
data.hist(column="SkinThickness")
plt.xlabel("SkinThickness",fontsize=15)
plt.ylabel("Frequency",fontsize=15)
plt.show()
#histogram of column insulin
fig=plt.figure(figsize=(17,10))
data.hist(column="Insulin")
plt.xlabel("Insulin",fontsize=15)
plt.ylabel("Frequency",fontsize=15)
plt.show()
#histogram of column BMI
fig=plt.figure(figsize=(17,10))
data.hist(column="BMI")
plt.xlabel("BMI",fontsize=15)
plt.ylabel("Frequency",fontsize=15)
plt.show()
#histogram of column DiabetesPedigreeFunction
fig=plt.figure(figsize=(17,10))
data.hist(column="DiabetesPedigreeFunction")
plt.xlabel("DiabetesPedigreeFunction",fontsize=15)
plt.ylabel("Frequency",fontsize=15)
plt.show()
#histogram of column age
fig=plt.figure(figsize=(17,10))
data.hist(column="Age")
plt.xlabel("Age",fontsize=15)
plt.ylabel("Frequency",fontsize=15)
plt.show()
#histogram of column outcome
fig=plt.figure(figsize=(17,10))
data.hist(column="Outcome")
plt.xlabel("Outcome",fontsize=15)
plt.ylabel("Frequency",fontsize=15)
plt.show()

#dropping false data
data.drop(data[data['Pregnancies'] < 0].index, inplace = True)
data.drop(data[data['Glucose'] == 0].index, inplace = True)
data.drop(data[data['BloodPressure'] <= 0].index, inplace = True)
data.drop(data[data['SkinThickness'] < 0].index, inplace = True)
data.drop(data[data['BMI'] <= 0].index, inplace = True)

#displaying new data frame
data

#displaying total number of null values
data.isnull().sum()

#droping the rows with null values
data.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)

#displaying modified data frame
data

#new histogram plots
#histogram of column Pregnancies
fig=plt.figure(figsize=(17,10))
data.hist(column="Pregnancies")
plt.xlabel("Pregnancies",fontsize=15)
plt.ylabel("Frequency",fontsize=15)
plt.show()
#histogram of column glucose
fig=plt.figure(figsize=(17,10))
data.hist(column="Glucose")
plt.xlabel("Glucose",fontsize=15)
plt.ylabel("Frequency",fontsize=15)
plt.show()
#histogram of column Blood Pressure
fig=plt.figure(figsize=(17,10))
data.hist(column="BloodPressure")
plt.xlabel("BloodPressure",fontsize=15)
plt.ylabel("Frequency",fontsize=15)
plt.show()
#histogram of Skin thickness
fig=plt.figure(figsize=(17,10))
data.hist(column="SkinThickness")
plt.xlabel("SkinThickness",fontsize=15)
plt.ylabel("Frequency",fontsize=15)
plt.show()
#histogram of column insulin
fig=plt.figure(figsize=(17,10))
data.hist(column="Insulin")
plt.xlabel("Insulin",fontsize=15)
plt.ylabel("Frequency",fontsize=15)
plt.show()
#histogram of column BMI
fig=plt.figure(figsize=(17,10))
data.hist(column="BMI")
plt.xlabel("BMI",fontsize=15)
plt.ylabel("Frequency",fontsize=15)
plt.show()
#histogram of column DiabetesPedigreeFunction
fig=plt.figure(figsize=(17,10))
data.hist(column="DiabetesPedigreeFunction")
plt.xlabel("DiabetesPedigreeFunction",fontsize=15)
plt.ylabel("Frequency",fontsize=15)
plt.show()
#histogram of column age
fig=plt.figure(figsize=(17,10))
data.hist(column="Age")
plt.xlabel("Age",fontsize=15)
plt.ylabel("Frequency",fontsize=15)
plt.show()
#histogram of column outcome
fig=plt.figure(figsize=(17,10))
data.hist(column="Outcome")
plt.xlabel("Outcome",fontsize=15)
plt.ylabel("Frequency",fontsize=15)
plt.show()

import seaborn as sns
sns.set(color_codes =True)

# Screening of Association between variables to study bivariate relationship
sns.pairplot(data, vars=["Pregnancies", "Glucose","BloodPressure","SkinThickness","Insulin", "BMI","DiabetesPedigreeFunction", "Age"],hue="Outcome")
plt.title("Pairplot of Variables by Outcome")

#Inference from Pair Plots
cor_l = data.corr(method ='pearson')
cor_l

sns.heatmap(cor_l)

#identifying diabetes with positive and negative results
num_obs = len(data)
num_true = len(data.loc[data['Outcome'] == 1])
num_false = len(data.loc[data['Outcome'] == 0])
print("Number of True cases:  {0} ({1:2.2f}%)".format(num_true, ((1.00 * num_true)/(1.0 * num_obs)) * 100))
print("Number of False cases: {0} ({1:2.2f}%)".format(num_false, (( 1.0 * num_false)/(1.0 * num_obs)) * 100))

#splitting into training set and test set
from sklearn.model_selection import train_test_split

feature_col_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
predicted_class_names = ['Outcome']

X = data[feature_col_names].values     # feature columns (8 X m)
y = data[predicted_class_names].values # class (1=true, 0=false) column (1 X m)
split_test_size = 0.25

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_test_size, random_state=29) 
                            # test_size = 0.25 is 25%

#displaying number of test and train sets
trainval = (1.0 * len(X_train)) / (1.0 * len(data.index))
testval = (1.0 * len(X_test)) / (1.0 * len(data.index))
print("{0:0.2f}% in training set".format(trainval * 100))
print("{0:0.2f}% in test set".format(testval * 100))

#importing model
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=1000, random_state=0)
rf.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(rf.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(rf.score(X_test, y_test)))

#feature importance plot
def plot_feature_importances_diabetes(model):
    diabetes_features = [x for i,x in enumerate(data.columns) if i!=8]
    plt.figure(figsize=(8,6))
    n_features = 8
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), diabetes_features)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)
plot_feature_importances_diabetes(rf)
plt.savefig('feature_importance')
