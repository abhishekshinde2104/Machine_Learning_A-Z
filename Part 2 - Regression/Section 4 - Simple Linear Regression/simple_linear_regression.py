#Step1:preprocess the data
import  numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('Salary_Data.csv')
#importing datasets
X=dataset.iloc[:,:-1].values #X=dataset.iloc[:,0].values this is independent variable
y=dataset.iloc[:,1].values#dependent variable vector

#splitting the dataset into Training set and Test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/3,random_state=0)

#feature scaling
"""from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)"""

#Step 2 : Training the set
#Fitting simple linaer regression to the training set
#library used to build the model will take care of feature scaling

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)
#here the machine is the regressor obj which we train using the trainig set to 
#understand the correlation between the training set variables

#Step 3 : Predicting the Test set results
y_pred=regressor.predict(X_test)

#Step 4 : Visualising the Training set Results
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Salary vs Experince(Training Set)')
plt.xlabel('Years of Exprience')
plt.ylabel('Salary')
plt.show()


#Step 4 : Visualising the Test set Results
plt.scatter(X_test,y_test,color='orange')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Salary vs Experince(Test Set)')
plt.xlabel('Years of Exprience')
plt.ylabel('Salary')
plt.show()


