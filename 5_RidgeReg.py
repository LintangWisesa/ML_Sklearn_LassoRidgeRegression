import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# data dummy
x = 10 * np.random.RandomState(1).rand(50)
x = np.sort(x)
# x = np.linspace(0, 10, 100)
print(x)
y = 2 * x - 5 + np.random.RandomState(1).randn(50)
# y = np.sin(x)
print(y)

# plot dummy data
plt.scatter(x, y)
# plt.show()

# ridge regression, polynomial, make_pipeline
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

model = make_pipeline(
    PolynomialFeatures(8),
    Ridge(alpha=1e-15)
)

model.fit(x.reshape(-1, 1), y)

# plot best fit line
plt.scatter(x, y)
plt.plot(x, model.predict(x.reshape(-1, 1)))
plt.show()