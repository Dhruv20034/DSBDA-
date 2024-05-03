import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv(r'E:\DSBDA\DSBDA Datasets\housing.csv')
print(data)

x=np.array([95,85,80,70,60])
y=np.array([85,95,70,65,70])
model=np.polyfit(x,y,1)
model
predict=np.poly1d(model)
predict(65)
y_pred=predict(x)
y_pred

from sklearn.metrics import r2_score
r2_score(y,y_pred)

y_line=model[1]+model[0]*x
plt.plot(x,y_line,c='r')
plt.scatter(x,y_pred)
plt.scatter(x, y,c='r')

from sklearn.datasets import fetch_california_housing
housing=fetch_california_housing()
data=pd.DataFrame(housing.data)
data.columns=housing.feature_names
data.head()
data['PRICE']=housing.target
data.isnull().sum()
x=data.drop(['PRICE'],axis=1)
y=data['PRICE']

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
lm=LinearRegression()
model=lm.fit(xtrain, ytrain)

ytrain_pred=lm.predict(xtrain)
ytest_pred=lm.predict(xtest)

df=pd.DataFrame(ytrain_pred,ytrain)
df=pd.DataFrame(ytest_pred,ytest)

from sklearn.metrics import mean_squared_error,r2_score
mse=mean_squared_error(ytest, ytest_pred)
print(mse)
mse=mean_squared_error(ytrain_pred,ytrain)
print(mse)
mse=mean_squared_error(ytest,ytest_pred)
print(mse)

plt.scatter(ytrain,ytrain_pred,c='blue',marker='o',label='Training data')
plt.scatter(ytest,ytest_pred,c='lightgreen',marker='s',label='Test data')
plt.xlabel('True values')
plt.ylabel('Predicted')
plt.title('True value vs Predicted value')
plt.legend(loc='upper left')
#plt.hlines(y=0,xmin=0,xmax=50)
plt.plot()
plt.show()
