### Loading and cleaning Data
import pandas as pd
DataFrame = pd.read_csv('.../LinearRegression/venv/Salary_dataset.csv')
X = DataFrame.iloc[:, :-1].values
y = DataFrame.iloc[:, 1].values

### Split data into training and testing
from sklearn.model_selection import train_test_split
X_train,  X_test, y_train, y_test = train_test_split(X,y, test_size =1/3, random_state=0)

### Fit Simple Linear Regression to Trainug Data
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)


### predict the test set
y_pred = regressor.predict(X_test) 
print(y_pred)
# Step 5 - Visualize training set results
import matplotlib.pyplot as plt
# plot the actual data points of training set
plt.scatter(X_train, y_train, color = 'red')
# plot the regression line
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

### make new prediction
new_salary = regressor.predict([[15]])
print("new prediction")
print(new_salary)
