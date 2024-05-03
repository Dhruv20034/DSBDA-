import pandas as pd
df=pd.read_csv(r'E:\DSBDA\DSBDA Datasets\Social_Network_Ads.csv')
print(df)

df['Gender']

df.isnull()

df.dtypes

df['Gender']=df['Gender'].map({'Male':1,'Female':0})
df['Gender']

x=df.drop(['Age'],axis=1)
y=df['Age']

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.25,random_state=0)
from sklearn.preprocessing import StandardScaler
st_x=StandardScaler()
xtrain=st_x.fit_transform(xtrain)
xtest=st_x.transform(xtest)

from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(xtrain, ytrain)

y_pred=classifier.predict(xtest)
y_pred

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(ytest,y_pred)
cm
