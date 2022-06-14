# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1.Import the required packages.<br>
2.Import the dataset to operate on.<br>
3.Split the dataset.<br>
4.Predict the required output.<br>
5.End the program.<br>

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Ranjith D
RegisterNumber: 212221240044
*/
```
~~~
import pandas as pd
data=pd.read_csv("spam.csv",encoding='latin-1')
data.head()
data.info()
data.isnull().sum()
x=data["v1"].values
y=data["v2"].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.feature_extractiaon.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
~~~
## Output:
Data Head:<br>
![SVM For Spam Mail Detection](https://github.com/RanjithD18/Implementation-of-SVM-For-Spam-Mail-Detection/blob/main/5.png)<br>Data Info:<br>
![SVM For Spam Mail Detection](https://github.com/RanjithD18/Implementation-of-SVM-For-Spam-Mail-Detection/blob/main/1.png)<br>Data isnull():<br>
![SVM For Spam Mail Detection](https://github.com/RanjithD18/Implementation-of-SVM-For-Spam-Mail-Detection/blob/main/2.png)<br>y_pred:<br>
![SVM For Spam Mail Detection](https://github.com/RanjithD18/Implementation-of-SVM-For-Spam-Mail-Detection/blob/main/3.png)<br>Accuracy:<br>
![SVM For Spam Mail Detection](https://github.com/RanjithD18/Implementation-of-SVM-For-Spam-Mail-Detection/blob/main/4.png)<br>
## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
