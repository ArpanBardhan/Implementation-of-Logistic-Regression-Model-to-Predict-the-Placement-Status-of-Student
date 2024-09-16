# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data.
2. Print the placement data and salary data.
3. Find the null and duplicate values.
4. Using logistic regression find the predicted values of accuracy , confusion matrices.
5. Display the results.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: ARPAN BARDHAN
RegisterNumber:  212222040013
*/
```

```python
import pandas as pd
data = pd.read_csv('/content/Placement_Data.csv')
data.head()

data1 = data.copy()
data1 = data1.drop(['sl_no','salary'],axis=1)
data1.head()
data1.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data1['gender'] = le.fit_transform(data1['gender'])
data1['ssc_b'] = le.fit_transform(data1['ssc_b'])
data1['hsc_b'] = le.fit_transform(data1['hsc_b'])
data1['hsc_s'] = le.fit_transform(data1['hsc_s'])
data1['degree_t'] = le.fit_transform(data1['degree_t'])
data1['workex'] = le.fit_transform(data1['workex'])
data1['specialisation'] = le.fit_transform(data1['specialisation'])
data1['status'] = le.fit_transform(data1['status'])

data1

x = data1.iloc[:,:-1]
x
y = data1['status']
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear")
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)

accuracy
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion

```

## Output:
![Screenshot 2024-09-16 101833](https://github.com/user-attachments/assets/82c9eba0-63dd-4593-b3f3-2503ad208844)

![Screenshot 2024-09-16 101839](https://github.com/user-attachments/assets/1dc77074-8d91-47ef-ada6-8f5d32eb470e)

![Screenshot 2024-09-16 101846](https://github.com/user-attachments/assets/d3b87b5a-3654-4678-a8a3-9cfb5eaf2fc2)



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
