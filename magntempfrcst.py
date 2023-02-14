#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 13:34:12 2023

@author: bazandavinci
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.color_palette("deep")
sns.set_theme()

df = pd.read_csv("~/Documents/Datasets/MotorTemp/measures_v2.csv")

"There was 17 total measure sessions, I'll be focusing mostly on some of them"

df_7 = df[df["profile_id"] == 7].reset_index()
plt.figure(figsize=(10, 8))
sns.heatmap(df_7.corr(), cmap = "viridis", annot = True)

"pm is the permanent magnet temperature, and stator_yoke is the temperature of the stator yoke"
"we can appreciate that the stator yoke temperature is heavily correlated with stator_winding"
"and stator tooth, maybe it would be necesarry that we emove them for our machine learning algorithm"
"motor speed has no correlation at all, we should remove it"
df_7['index_column'] = df_7.index

plt.figure()
sns.lineplot(data = df_7, x="index_column", y="pm")

plt.figure()
sns.lineplot(data = df_7, x="index_column", y="stator_yoke")

"the stator yoke variable between 2200 and 12000 looks like a good candidate for forecasting"

stats = df.describe()
stats7 = df_7.describe()

plt.figure()
sns.histplot(x = df["torque"])

plt.figure()
sns.histplot(x = df["motor_speed"])

"we can see that the torque from the 7th session is efficient, it has high torque but the motor speed is probably"
"in the first or second decile"
"also it doesnt gets as hot as other sessions, so this will be a forecast of a good session"

X = df_7.drop(columns = ["stator_yoke", "motor_speed"])
y = df_7["stator_yoke"]

X_train = X[X["index_column"] < 9000]
y_train = y.iloc[0:9000]

X_test = X.iloc[9000:12100]
y_test = y.iloc[9000:12100]

from xgboost import XGBRegressor

from sklearn.model_selection import GridSearchCV,  KFold
from sklearn.metrics import mean_squared_error

xgb = XGBRegressor(verbosity = 1, n_estimators=300)

kf = KFold(5) 

params = { "learning_rate": [.00001, .0001, .001, .01, .1], 
           "min_split_loss": [1, 35, 100],
           "max_depth": [3, 5, 8],
           "reg_lambda": [1, 3, 10],
           "reg_alpha": [.001, .01, .1, 1, 3, 10],
           }

model = GridSearchCV(xgb, param_grid = params, cv=kf)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

score = model.score(X_test, y_test)
msre = mean_squared_error(y_test, y_pred)

plt.figure()
plt.plot(X_test["index_column"], y_test, color = "m")
plt.plot(X_test["index_column"], y_pred, color = "c")
plt.title("Forecasting")
plt.xlabel("Timelapse")
plt.ylabel("Permanent Magnet Temperature")
plt.legend(["Actual Temp", "Forecasted Temp"])
plt.show()