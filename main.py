import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import metrics
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split

df = pd.read_csv('./data/Video_Games.csv')

print(df.describe())

df_res = pd.read_csv('./data/Video_Games_Test.csv')

#print(df.head())

#print('\nКоличество строк и количество столбцов: ', df.shape[0], df.shape[1])

#print('\nОсновная статистика по датасету:\n', df.describe())

#print('\nПроверяем количество пропусков:\n', df.isnull().sum())

df_dummies = pd.get_dummies(df, columns=['Publisher', 'Developer', 'Platform', 'Genre', 'Rating'], drop_first=False).iloc[:, 1:]

df_res_dummies = pd.get_dummies(df_res, columns=['Publisher', 'Developer', 'Platform', 'Genre', 'Rating'], drop_first=True).iloc[:, 2:]

df_dummies.drop(columns=['User_Count', 'Critic_Count'], inplace=True)

df_res_dummies.drop(columns=['User_Count', 'Critic_Count'], inplace=True)

#print('\nСтало столбцов: ', df_dummies.shape[1])

for x in df_dummies.columns:
    if x not in df_res_dummies.columns and x != 'JP_Sales':
        del df_dummies[x]
        continue
    if df_dummies[x].notna().sum() < 0.8 * len(df_dummies):
        del df_dummies[x]

df_dummies.columns = df_dummies.columns.str.lower().str.replace(' ', '_')

#print('\nПроверяем количество пропусков:\n', df_dummies.isnull().sum())

imputer = KNNImputer(n_neighbors=8)
df_fill = imputer.fit_transform(df_dummies.values)

df_new = pd.DataFrame(df_fill, columns=df_dummies.columns)

clf = IsolationForest(random_state=0)
preds = clf.fit_predict(df_new)

i = 0
print('Выбросы:')
for x in preds:
    if x != 1:
        print(i, x)
    i += 1
#print(df_new.dtypes)

#print('\nПроверяем количество пропусков:\n', df_new.isnull().sum())

y_ = df_new['jp_sales'].values
X_ = df_new.drop(columns = ['jp_sales'], axis = 1).values

X_train_, X_test_, y_train_, y_test_ = train_test_split(X_, y_, test_size=0.2, random_state=0)
#print('Размер обучающей выборки {},\nРазмер тестовой выборки {} \n'.format(len(X_train), len(X_test)))

X_train_ = X_train_[ y_train_ > 0 ]
y_train_ = y_train_[ y_train_ > 0 ]

lab_enc = preprocessing.LabelEncoder()
encoded = lab_enc.fit_transform(y_train_)

feat_labels = df_new.columns
forest = RandomForestClassifier(n_estimators=500, random_state=1)

forest.fit(X_train_, encoded)

importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]

fields_for_del = []

for f in range(X_train_.shape[1]):
    if importances[indices[f]] >= 0:
        print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))
    else:
        fields_for_del.append(feat_labels[indices[f]])

df_new.drop(columns=fields_for_del, inplace=True)

corr_res = df_new.corr()

columns_for_del = []
i = 0
for x in corr_res.iloc[3]:
    if x < 0.1:
       columns_for_del.append(i)
    i += 1

columns_for_del_from_res = []
for x in columns_for_del:
    try:
        if df_new.columns[x] not in ['year_of_release']:
            columns_for_del_from_res.append(df_new.columns[x])
            df_new.drop(df_new.columns[x], axis=1, inplace=True)
    except:
        continue

print(df_new.shape)


#plt.figure(figsize = (15,8))
#sns.heatmap(df_new.corr(), annot = True, vmin=-1, vmax=1, center=0, cmap='coolwarm')
#plt.show()

y = df_new['jp_sales'].values
X = df_new.drop(columns = ['jp_sales'], axis = 1).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

X_train = X_train[ y_train > 0 ]
y_train = y_train[ y_train > 0 ]

#mms = preprocessing.MinMaxScaler()
#X_train_norm = mms.fit_transform(X_train)
#X_test_norm = mms.fit_transform(X_test)

#stdsc = preprocessing.StandardScaler()
#X_train_norm = stdsc.fit_transform(X_train)
#X_test_norm = stdsc.fit_transform(X_test)

#lab_enc = preprocessing.LabelEncoder()
#encoded = lab_enc.fit_transform(y_train)

 #model = LinearRegression().fit(X_train, y_train)

model = XGBRegressor(max_depth=20, n_estimators=500, learning_rate=0.1, random_state=0)
model.fit(X_train, y_train)

#model = RandomForestRegressor(max_depth=20, random_state=0).fit(X_train, y_train)

y_pred = model.predict(X_test)

print(y_pred)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

df_res_dummies.columns = df_res_dummies.columns.str.lower().str.replace(' ', '_')

for x in df_res_dummies.columns:
    if x not in df_dummies.columns:
        del df_res_dummies[x]
        continue
    if df_res_dummies[x].notna().sum() < 0.8 * len(df_res_dummies):
        del df_res_dummies[x]

imputer = KNNImputer(n_neighbors=8)
df_res_fill = imputer.fit_transform(df_res_dummies.values)

df_res_new = pd.DataFrame(df_res_fill, columns=df_res_dummies.columns)

df_res_new.drop(columns=fields_for_del, axis=1, inplace=True)

df_res_new.drop(columns=columns_for_del_from_res, axis=1, inplace=True)

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