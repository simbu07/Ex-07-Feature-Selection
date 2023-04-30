# Ex-07-Feature-Selection
## AIM
To Perform the various feature selection techniques on a dataset and save the data to a file. 

# Explanation
Feature selection is to find the best set of features that allows one to build useful models.
Selecting the best features helps the model to perform well. 

# ALGORITHM
### STEP 1
Read the given Data
### STEP 2
Clean the Data Set using Data Cleaning Process
### STEP 3
Apply Feature selection techniques to all the features of the data set
### STEP 4
Save the data to the file


# PROGRAM:
```
Developed By: Silambarasan K
Reg No:212221230101
```
```py
#importing library
import pandas as pd
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# data loading
data = pd.read_csv('titanic_dataset.csv')
data
data.tail()
data.isnull().sum()
data.describe()

#now, we are checking start with a pairplot, and check for missing values
sns.heatmap(data.isnull(),cbar=False)

#Data Cleaning and Data Drop Process
data['Fare'] = data['Fare'].fillna(data['Fare'].dropna().median())
data['Age'] = data['Age'].fillna(data['Age'].dropna().median())

# Change to categoric column to numeric
data.loc[data['Sex']=='male','Sex']=0
data.loc[data['Sex']=='female','Sex']=1

# instead of nan values
data['Embarked']=data['Embarked'].fillna('S') 

# Change to categoric column to numeric
data.loc[data['Embarked']=='S','Embarked']=0
data.loc[data['Embarked']=='C','Embarked']=1
data.loc[data['Embarked']=='Q','Embarked']=2

#Drop unnecessary columns
drop_elements = ['Name','Cabin','Ticket']
data = data.drop(drop_elements, axis=1)

data.head(11)

#heatmap for train dataset
f,ax = plt.subplots(figsize=(5, 5))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

# Now, data is clean and read to a analyze
sns.heatmap(data.isnull(),cbar=False)

# how many people survived or not... %60 percent died %40 percent survived
fig = plt.figure(figsize=(18,6))
data.Survived.value_counts(normalize=True).plot(kind='bar',alpha=0.5)
plt.show()

#Age with survived
plt.scatter(data.Survived, data.Age, alpha=0.1)
plt.title("Age with Survived")
plt.show()

#Count the pessenger class
fig = plt.figure(figsize=(18,6))
data.Pclass.value_counts(normalize=True).plot(kind='bar',alpha=0.5)
plt.show()

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

X = data.drop("Survived",axis=1)
y = data["Survived"]

mdlsel = SelectKBest(chi2, k=5) 
mdlsel.fit(X,y)
ix = mdlsel.get_support() 
data2 = pd.DataFrame(mdlsel.transform(X), columns = X.columns.values[ix]) # en iyi leri aldi... 7 tane...
data2.head(11)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

target = data['Survived'].values
data_features_names = ['Pclass','Sex','SibSp','Parch','Fare','Embarked','Age']
features = data[data_features_names].values

#Build test and training test
X_train,X_test,y_train,y_test = train_test_split(features,target,test_size=0.3,random_state=42)

my_forest = RandomForestClassifier(max_depth=5, min_samples_split=10, n_estimators=500, random_state=5,criterion = 'entropy')


my_forest_ = my_forest.fit(X_train,y_train)
target_predict=my_forest_.predict(X_test)

print("Random forest score: ",accuracy_score(y_test,target_predict))

from sklearn.metrics import mean_squared_error, r2_score
print ("MSE    :",mean_squared_error(y_test,target_predict))
print ("R2     :",r2_score(y_test,target_predict))
```
# OUPUT:
### Dataset:
![Dataset](https://user-images.githubusercontent.com/94525786/235351611-43c1d068-0709-4fbf-8b6b-18b2de8316a1.png)

### Null Values:
![null](https://user-images.githubusercontent.com/94525786/235351619-f55d5efd-501f-43e6-990d-1cc9659bbed6.png)

### Describe:
![describe](https://user-images.githubusercontent.com/94525786/235351621-a92e1da3-10c2-42cd-b0af-e0c611ce8694.png)

### missing values::
![Missingvalue](https://user-images.githubusercontent.com/94525786/235351625-65cdf214-7952-4968-ad7c-6610974d7ebd.png)

### Data after cleaning:
![Cleaneddata](https://user-images.githubusercontent.com/94525786/235351633-073a22ba-3c47-4133-85d4-c681e03a0c35.png)

### Data on Heatmap:
![Heatmap](https://user-images.githubusercontent.com/94525786/235351639-bb34ace7-c70a-43bf-8ff1-7dca4d41b725.png)

### Cleaned Null values:
![After_cleaned_null](https://user-images.githubusercontent.com/94525786/235351652-e1b2e383-6355-4487-9c07-58de31302e76.png)

### Report of (people survived & Died):
![surived](https://user-images.githubusercontent.com/94525786/235351659-019b0b7c-2b73-47f4-97b6-fa9706eac6cf.png)
### Report of Survived People's Age:
![age_survived](https://user-images.githubusercontent.com/94525786/235351710-0b936617-f531-4f71-b891-f72d583725a8.png)

### Report of pessengers:
![Count_passenger](https://user-images.githubusercontent.com/94525786/235351692-d8157671-3066-486d-ad32-59fd2812edef.png)

### Report:
![out](https://user-images.githubusercontent.com/94525786/235351718-94c2066b-30db-4226-9628-90dedb731ff7.png)

## RESULT:
Thus, Sucessfully performed the various feature selection techniques on a given dataset.
