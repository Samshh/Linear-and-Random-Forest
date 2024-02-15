import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import numpy as np

data_file_url = input("Enter the URL of the dataset: ")
target_column_index = int(input("Enter the index of the target variable (starting from 0): "))
data_frame = pd.read_csv(data_file_url)

y = data_frame.iloc[:, target_column_index]
x = data_frame.drop(data_frame.columns[target_column_index], axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

lr = LinearRegression()
lr.fit(x_train, y_train)

y_lr_train_pred = lr.predict(x_train)
y_lr_test_pred = lr.predict(x_test)

lr_train_mse = mean_squared_error(y_train, y_lr_train_pred)
lr_train_r2 = r2_score(y_train, y_lr_train_pred)

lr_test_mse = mean_squared_error(y_test, y_lr_test_pred)
lr_test_r2 = r2_score(y_test, y_lr_test_pred)

lr_results = pd.DataFrame(['Linear Regression', lr_train_mse, lr_train_r2, lr_test_mse, lr_test_r2]).transpose()
lr_results.columns = ['Method', 'Train MSE', 'Train R2', 'Test MSE', 'Test R2']

rf = RandomForestRegressor(max_depth=2, random_state=100)
rf.fit(x_train, y_train)

y_rf_train_pred = rf.predict(x_train)
y_rf_test_pred = rf.predict(x_test)

rf_train_mse = mean_squared_error(y_train, y_rf_train_pred)
rf_train_r2 = r2_score(y_train, y_rf_train_pred)

rf_test_mse = mean_squared_error(y_test, y_rf_test_pred)
rf_test_r2 = r2_score(y_test, y_rf_test_pred)

rf_results = pd.DataFrame(['Random Forest', rf_train_mse, rf_train_r2, rf_test_mse, rf_test_r2]).transpose()
rf_results.columns = ['Method', 'Train MSE', 'Train R2', 'Test MSE', 'Test R2']

df_models = pd.concat([lr_results, rf_results], axis=0).reset_index(drop=True)
print(df_models)

z = np.polyfit(y_train, y_lr_train_pred, 1)
p = np.poly1d(z)

plt.figure(figsize=(5, 5))
plt.scatter(x=y_train, y=y_lr_train_pred, alpha=0.3, label='Training data')
plt.plot(y_train, p(y_train), '#FFA500', label='Trendline')
plt.xlabel('Experimental')
plt.ylabel('Predicted')
plt.title('Linear Regression: Predicted vs Experimental')
plt.legend(loc='upper left')
plt.show()