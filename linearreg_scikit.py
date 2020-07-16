import pandas as pd
import numpy as np
from sklearn.metrics  import r2_score
from sklearn.linear_model import LinearRegression
import os

train = pd.read_csv('.../traincsv.csv')
test = pd.read_csv('.../testcsv.csv')
train = train.dropna()
x_train = train['x']
y_train = train['y']
x_test = test['x']
y_test = test['y']


x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)


x_train = x_train.reshape(-1,1)
x_test = x_test.reshape(-1,1)


clf = LinearRegression(normalize = True)
clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)
print(r2_score(y_test,y_pred))
