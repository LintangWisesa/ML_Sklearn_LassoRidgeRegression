import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# data dummy
# x = 10 * np.random.RandomState(1).rand(50)
x = np.linspace(0, 10, 100)
print(x)
# y = 2 * x - 5 + np.random.RandomState(1).randn(50)
y = np.sin(x)
print(y)

# plot dummy data
plt.scatter(x, y)
# plt.show()

# linear regression
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x.reshape(-1, 1), y)

# plot best fit line
plt.scatter(x, y)
plt.plot(x, model.predict(x.reshape(-1, 1)))
plt.show()