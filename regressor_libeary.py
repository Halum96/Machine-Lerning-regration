#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 20:10:21 2018

@author: arijitbag
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

"""
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)"""

# create regressor here

y_pred = regressor.predict(6.5)


plt.scatter(x, y, color = 'blue')
plt.plot(x, regressor.predict(x), color = 'red')
plt.titel('Truth vs bluff')
plt.xlabel('position level')
plt.ylabel('Salary')
plt.show()

# fro hd
x_grid = np.arrange(min(x), max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid, 1)))
plt.scatter(x, y, color = 'blue')
plt.plot(x_grid, regressor.predict(x_grid), color = 'red')
plt.titel('Truth vs bluff')
plt.xlabel('position level')
plt.ylabel('Salary')
plt.show()