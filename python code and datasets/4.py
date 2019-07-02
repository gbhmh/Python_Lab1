import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

advv = pd.read_csv('advertising.csv')

print(advv.columns)

#converting non-numeric features to numeric features
advv['Country'] =advv['Country'].astype('category').cat.codes
advv['Ad Topic Line'] =advv['Ad Topic Line'].astype('category').cat.codes
advv['City'] =advv['City'].astype('category').cat.codes

#checking for nulls in the data
nulls = pd.DataFrame(advv.isnull().sum().sort_values(ascending=False))
nulls.columns = ['Null Count']
nulls.index.name  = 'Feature'
# print(nulls)

#removing nulls in data
adv = advv.select_dtypes(include=[np.number]).interpolate().dropna()
print(sum(adv.isnull().sum()  != 0))


print(adv.head())

df = pd.DataFrame(adv)

# checking correlation of all the columns against target column
print(' correlation of all columns are :\n' + str(df[df.columns[:]].corr()['Clicked on Ad'].sort_values(ascending=False)))

# Taking all columns for analysis
X = adv.drop('Clicked on Ad',axis=1)
y = adv['Clicked on Ad']

# taking top correlated columns for analysis
A = adv[['Age','Ad Topic Line','Country']]
b = adv['Clicked on Ad']

# splitting data into train data for training model and test data for testing model
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=.2)
A_train, A_test, b_train, b_test = train_test_split(A, b, random_state=42, test_size=.2)

# fitting model to our train data
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)

lm2 = LinearRegression()
lm2.fit(A_train,b_train)

# print(lm.intercept_)

coeff = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
# print(coeff)

# testing the model comparing with test data
predictions = lm.predict(X_test)
predictions1 = lm2.predict(A_test)
plt.scatter(y_test,predictions)
# plt.show()

# metrics
from sklearn.metrics import r2_score
print('total r2 score is ',r2_score(y_test,predictions))
print('correlated r2 score is ',r2_score(b_test,predictions1))
# print('r2 score is ',lm.score(X_test,y_test))
# print('r2 score is ',lm.score(y_test,predictions))
from sklearn.metrics import mean_squared_error
print('total rmse',mean_squared_error(y_test,predictions))
print('correlated rmse',mean_squared_error(b_test,predictions1))