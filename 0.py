import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# data dummy
x = 10 * np.random.RandomState(1).rand(50)
print(x)
y = 2 * x - 5 + np.random.RandomState(1).randn(50)
print(y)

# plot dummy data
plt.scatter(x, y)
plt.show()