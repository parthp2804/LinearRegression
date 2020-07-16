import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
House = pd.read_csv('C:/Users/Parth/PycharmProjects/LinearRegression/venv/Housing.csv')


# visualizing the distribution of housing prices
sns.distplot(House['Price'],hist_kws=dict(edgecolor='black', linewidth=1),color='Blue')


# X and y

columns_to_keep = ['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms','Avg. Area Number of Bedrooms', 'Area Population']
X = House[columns_to_keep]

y = House['Price']

# train and test data

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4,random_state=101)

# Fit the model
from sklearn.linear_model import LinearRegression
HousingPrice = LinearRegression()
HousingPrice.fit(X_train, y_train)

# evaluate the model by checking  the coeffecients and intercept
intercept = HousingPrice.intercept_
print('Intercept:', intercept)

co = pd.DataFrame(HousingPrice.coef_, X.columns, columns=['Coeff'])
print('coeffecients')
print(co)

# Predict the Price
price_pred = HousingPrice.predict(X_test)
print(price_pred)

# visualize the predicted price and test price

plt.scatter(y_test,price_pred,edgecolor="black")
plt.show()