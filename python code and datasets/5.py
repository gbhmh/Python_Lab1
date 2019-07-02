from sklearn.datasets import load_diabetes
from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()
import pandas as pd
import numpy as np
df = pd.read_csv('Loan payments data.csv')
print(df.info())

#converting non-numeric features to numeric features using label encoder
df['loan_status']=le.fit_transform(df['loan_status'])
df['Gender']=le.fit_transform(df['Gender'])
df['education']=le.fit_transform(df['education'])
df['past_due_days']=le.fit_transform(df['past_due_days'])

print(df.head())
#checking for nulls in the data
nulls = pd.DataFrame(df.isnull().sum().sort_values(ascending=False))
nulls.columns = ['Null Count']
nulls.index.name  = 'Feature'
# print(nulls)

#removing nulls in data
df1 = df.select_dtypes(include=[np.number]).interpolate().dropna()
print(sum(df1.isnull().sum()  != 0))

# checking correlation of all the columns against target column
print(' correlation of all columns are :\n' + str(df1[df1.columns[:]].corr()['loan_status'].sort_values(ascending=False)))

# taking top correlated columns for analysis
A = df1[['past_due_days','age','education']]
# taking target column
b = df1['loan_status']

# splitting data into train data for training model and test data for testing model
from sklearn.model_selection import train_test_split
A_train, A_test, b_train, b_test = train_test_split(A, b, random_state=42, test_size=.1)

# using naive bayes model
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()

#using svm
from sklearn.svm import SVC
model = SVC()

#using knn
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()


# fitting model to our train data
gnb.fit(A_train,b_train)
model.fit(A_train,b_train)
knn.fit(A_train,b_train)

# testing the model comparing with test data
predictions_naive = gnb.predict(A_test)
predictions_svm = model.predict(A_test)
predictions_knn = knn.predict(A_test)


# calculating root mean sqaure value of the errors
from sklearn.metrics import mean_squared_error
print ('RMSE for naive bayes is: \n', mean_squared_error(b_test, predictions_naive))
print ('RMSE for svm is: \n', mean_squared_error(b_test, predictions_svm))
print ('RMSE for knn is: \n', mean_squared_error(b_test, predictions_knn))


from sklearn import metrics
print("Accuracy for naive bayes is : ",round(metrics.accuracy_score(b_test, predictions_naive) * 100, 2))
# print("classification_report\n",metrics.classification_report(y_test,predictions_naive))
# print("confusion matrix\n",metrics.confusion_matrix(y_test,predictions_naive))
print("Accuracy for svm is : ",round(metrics.accuracy_score(b_test, predictions_svm) * 100, 2))
# print("classification_report\n",metrics.classification_report(y_test,predictions_svm))
# print("confusion matrix\n",metrics.confusion_matrix(y_test,predictions_svm))

print("Accuracy for knn is : ",round(metrics.accuracy_score(b_test, predictions_knn) * 100, 2))
# print("classification_report\n",metrics.classification_report(y_test,predictions_knn))
# print("confusion matrix\n",metrics.confusion_matrix(y_test,predictions_knn))

