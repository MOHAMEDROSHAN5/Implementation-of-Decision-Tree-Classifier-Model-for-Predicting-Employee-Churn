# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Upload and read the dataset.
3. Check for any null values using the isnull() function.
4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5. Find the accuracy of the model and predict the required values by importing the required module fr

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: MOHAMED ROSHAN S
RegisterNumber:  212222040101
*/
```

```py
import pandas as pd
data=pd.read_csv('Employee.csv')

data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data["salary"]=le.fit_transform(data["salary"])

data.head()

data["Departments "]=le.fit_transform(data["Departments "])

data.head(100)

data.info()

x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","Departments ","salary"]]
x.head()

x.info()

y=data[["left"]]

y.head()

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

x_train.shape
x_test.shape
y_train.shape
y_test.shape

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
print(accuracy)

dt.predict([[0.9,9,8,9,268,6,0,1,2]])
```

## Output:
## DATA HEAD
![DATA](Screenshot%202023-11-08%20202357.png)
## DATA INFO
![DATA](Screenshot%202023-11-08%20202408.png)
## DNULL DATASET
![DATA](Screenshot%202023-11-08%20202419.png)
## VALUES COUNT IN LEFT COLUMN
![DATA](Screenshot%202023-11-08%20202429.png)
## DATA TRANSFORMED HEAD
![DATA](Screenshot%202023-11-08%20203952.png)
## X HEAD
![DATA](Screenshot%202023-11-08%20202607.png)
## ACCURACY
![DATA](Screenshot%202023-11-08%20202620.png)
## DATA PREDICTION
![DATA](Screenshot%202023-11-08%20202625.png)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
