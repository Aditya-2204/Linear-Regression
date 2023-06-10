import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

x_train = [1,2,3,4,5,6,7,8,9,10]
y_train = [23,67,48,21,64,9,11,100,24,52]

x_train = np.reshape(x_train, (-1,1))
y_train = np.reshape(y_train, (-1,1))

lr = LinearRegression()

lr.fit(x_train, y_train)

m = float(lr.coef_)
lr.intercept_ = np.reshape(lr.intercept_, (-1,1))
c = float(lr.intercept_)

#creating the linear equation
x = np.linspace(0, 20, 100)
y = m*x+c



fig = plt.figure(figsize = (10, 5))
# Create the plot
plt.plot(x,y)
plt.scatter(x_train, y_train)

 
# Show the plot
plt.show()