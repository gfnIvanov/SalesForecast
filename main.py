import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

df = pd.read_csv('./data/Video_Games.csv')

df_res = pd.read_csv('./data/Video_Games_Test.csv')

#print(df.head())

#print('\nКоличество строк и количество столбцов: ', df.shape[0], df.shape[1])

#print('\nОсновная статистика по датасету:\n', df.describe())

#print('\nПроверяем количество пропусков:\n', df.isnull().sum())

df_dummies = pd.get_dummies(df, columns=['Publisher', 'Developer', 'Platform', 'Genre', 'Rating'], drop_first=False).iloc[:, 1:]

df_res_dummies = pd.get_dummies(df_res, columns=['Publisher', 'Developer', 'Platform', 'Genre', 'Rating'], drop_first=True).iloc[:, 2:]

#print('\nСтало столбцов: ', df_dummies.shape[1])

for x in df_dummies.columns:
    if x not in df_res_dummies.columns and x != 'JP_Sales':
        del df_dummies[x]
        continue
    if df_dummies[x].notna().sum() < 0.8 * len(df_dummies):
        del df_dummies[x]


#print('\nПроверяем количество пропусков:\n', df_dummies.isnull().sum())

imputer = KNNImputer(n_neighbors=8)
df_fill = imputer.fit_transform(df_dummies.values)

df_new = pd.DataFrame(df_fill, columns=df_dummies.columns)

#print(df_new.dtypes)

#print('\nПроверяем количество пропусков:\n', df_new.isnull().sum())

y = df_new['JP_Sales'].values
del df_new['JP_Sales']
X = df_new.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#print('Размер обучающей выборки {},\nРазмер тестовой выборки {} \n'.format(len(X_train), len(X_test)))

X_train = X_train[ y_train > 0 ]
y_train = y_train[ y_train > 0 ]

#model = LinearRegression().fit(X_train, y_train)

model = XGBRegressor(max_depth=20, n_estimators=500, learning_rate=0.1, random_state=0)
model.fit(X_train, y_train)

#model = RandomForestRegressor(max_depth=20, random_state=0).fit(X_train, y_train)

y_pred = model.predict(X_test)

print(y_pred)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


for x in df_res_dummies.columns:
    if x not in df_dummies.columns:
        del df_res_dummies[x]
        continue
    if df_res_dummies[x].notna().sum() < 0.8 * len(df_res_dummies):
        del df_res_dummies[x]

imputer = KNNImputer(n_neighbors=8)
df_res_fill = imputer.fit_transform(df_res_dummies.values)

df_res_new = pd.DataFrame(df_res_fill, columns=df_res_dummies.columns)

#print(df_res_new.dtypes)

X_res_test = df_res_new.values

res_predictions = model.predict(X_res_test)

res_predictions = np.around(res_predictions, decimals = 2)

result_list = []
i = 1

for x in res_predictions:
    result_list.append(str(i) + ',' + str(x))
    i += 1

result = pd.DataFrame(result_list)

result.to_csv('./data/Video_Games_Results.csv', index=False)