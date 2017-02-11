import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

df = pd.read_csv('challenge_dataset.txt',header=None)

x_values = df[[0]]
y_values = df[[1]]

reg_model = linear_model.LinearRegression()
reg_model.fit(x_values, y_values)

#pred = reg_model.predict(10.274)

plt.scatter(x_values, y_values)
plt.plot(x_values, reg_model.predict(x_values))
plt.show()
