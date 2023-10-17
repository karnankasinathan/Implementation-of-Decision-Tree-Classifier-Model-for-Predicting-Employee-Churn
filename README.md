# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. import the required libraries.
2. Upload and read the dataset.
3. Check for any null values using the isnull() function.
4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5. Find the accuracy of the model and predict the required values by importing the required module from sklearn.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: karnan k
RegisterNumber:  212222230062
*/

import pandas as pd
data=pd.read_csv("Employee.csv")
data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder 
le=LabelEncoder()

data["salary"]=le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=data["left"]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test, y_pred)
accuracy

dt.predict([[0.5, 0.8, 9, 260, 6, 0, 1, 2]])
```

## Output:
### data.head()
![image](https://github.com/karnankasinathan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118787064/fd68c75a-3a41-44f3-afd4-7b39c3293290)

### data.info()
![image](https://github.com/karnankasinathan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118787064/e1add0ee-39e5-4a64-9c42-ea5fe387eb58)

### isnull() and sum()
![image](https://github.com/karnankasinathan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118787064/068f877d-8b9e-4549-8606-10808f678023)

### data value counts()
![image](https://github.com/karnankasinathan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118787064/e853d094-befa-4c52-a842-8bfd82c36655)

### data.head() for salary
![image](https://github.com/karnankasinathan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118787064/88a4c9a3-c1d9-4e33-b97a-7480ae53bc29)

### x.head()
![image](https://github.com/karnankasinathan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118787064/e7ee3b5d-4a56-4597-acf7-0fa519f82913)

### accuracy value
![image](https://github.com/karnankasinathan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118787064/bbc5ce06-16d0-4c96-8cb3-1b08fb2da074)

### data prediction
![image](https://github.com/karnankasinathan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118787064/460a7424-4b46-4bab-95a6-dfc53ec5d0de)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
