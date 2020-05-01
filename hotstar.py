# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


df=pd.read_csv('mediacompany.csv')
df.head()

df['Date']=pd.to_datetime(df['Date'])

from datetime import datetime
d0=datetime(2017,2,28)
d1=df.Date
delta=d1-d0
df['days']=delta

df['days']=df['days'].astype(str)
df['days']=df['days'].map(lambda x:x[0:2])
df['days']=df['days'].astype(int)

df['weekdays']=(df['days']+3)%7
df['weekdays'].replace(0,7,inplace=True)
df['Ad_impression']=df['Ad_impression']/1000

df['weekends']=df['weekdays']
df['weekends'].replace(2,0,inplace=True)
df['weekends'].replace(3,0,inplace=True)
df['weekends'].replace(4,0,inplace=True)
df['weekends'].replace(5,0,inplace=True)
df['weekends'].replace(6,0,inplace=True)
df['weekends'].replace(7,1,inplace=True)

X=df[['Views_platform','weekends','Ad_impression']]
y=df['Views_show']

import statsmodels.api as sm

x=sm.add_constant(X)
model=sm.OLS(y,X).fit()

print(model.summary())

y_pred=model.predict(X)

sns.heatmap(df.corr(),annot=True)

from sklearn.metrics import r2_score,mean_squared_error

print(r2_score(df['Views_show'], y_pred))
print(mean_squared_error(df['Views_show'], y_pred))

c=[i for i in range(1,81,1)]

plt.plot(c,df.Views_show,color='blue')
plt.plot(c,y_pred,color='orange')
plt.show()

y_err=df.Views_show-y_pred


plt.plot(c,y_err,color='blue')
plt.show()