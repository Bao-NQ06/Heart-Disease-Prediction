import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, StackingClassifier
from xgboost import XGBClassifier

df = pd.read_csv('cleveland.csv', header = None)
df.columns = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','target']
df['target'] = df.target.map({0: 0 , 1: 1 , 2: 1 , 3: 1 , 4: 1})
df['thal'] = df.thal.fillna(df.thal.mean())
df['ca'] = df.ca.fillna(df.ca.mean())

distribution of target vs age
plt.figure(figsize=(15, 6))
sns.countplot(x='age', data=df, hue='target')
plt.title('Distribution of Age vs Target')
plt.xlabel('Age')
plt.ylabel('Count')
plt.legend(title='Target', loc='upper right')
plt.show()

barplot of age vs sex with hue = target
plt.figure(figsize=(15, 6))
sns.barplot(x='sex', y='age', data=df, hue='target')
plt.title('barplot of age vs sex with hue = target')
plt.xlabel('Sex')
plt.ylabel('Age')
plt.legend(title='Target', loc='upper right')
plt.show()

x= df.drop('target',axis=1)
y = df['target']


X_train , X_test , y_train , y_test = train_test_split(x , y , test_size = 0.2,random_state = 42)


model = KNeighborsClassifier(n_neighbors=5,weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
cm_train = confusion_matrix(y_train, model.predict(X_train))
cm_test = confusion_matrix(y_test, y_pred)
accuracy_for_train = np.round((cm_train [0][0] + cm_train [1][1]) /len(y_train) ,2)
accuracy_for_test = np.round((cm_test [0][0] + cm_test [1][1]) /len(y_test) ,2)
print('Accuracy for training set for KNeighborsClassifier = {}'. format(accuracy_for_train))
print('Accuracy for test set for KNeighborsClassifier = {}'. format(accuracy_for_test))


svm_model = SVC(kernel = 'rbf',random_state=42)
svm_model.fit(X_train, y_train)
svm_y_pred = svm_model.predict(X_test)
svm_cm_train = confusion_matrix(y_train, svm_model.predict(X_train))
svm_cm_test = confusion_matrix(y_test, svm_y_pred)
accuracy_for_train = np.round((svm_cm_train[0][0] + svm_cm_train[1][1]) /len(y_train) ,2)
accuracy_for_test = np.round((svm_cm_test[0][0] + svm_cm_test[1][1]) /len(y_test) ,2)
print('Accuracy for training set for SVM = {} '.format(accuracy_for_train))
print('Accuracy for test set for SVM = {} '.format(accuracy_for_test))


NB_model = GaussianNB()
NB_model.fit(X_train, y_train)
NB_y_pred = NB_model.predict(X_test)
NB_cm_train = confusion_matrix(y_train, NB_model.predict(X_train))
NB_cm_test = confusion_matrix(y_test, NB_y_pred)
accuracy_for_train = np.round((NB_cm_train[0][0] + NB_cm_train[1][1]) /len(y_train) ,2)
accuracy_for_test = np.round((NB_cm_test[0][0] + NB_cm_test[1][1]) /len(y_test) ,2)
print('Accuracy for training set for Naive Bayes = {} '.format(accuracy_for_train))
print('Accuracy for test set for Naive Bayes = {} '.format(accuracy_for_test))

DT_model = DecisionTreeClassifier(criterion='gini', max_depth=10, min_samples_split=2 ,random_state=42) 
DT_model.fit(X_train, y_train)
DT_y_pred = DT_model.predict(X_test)
DT_cm_train = confusion_matrix(y_train, DT_model.predict(X_train))
DT_cm_test = confusion_matrix(y_test, DT_y_pred)
accuracy_for_train = np.round((DT_cm_train[0][0] + DT_cm_train[1][1]) /len(y_train) ,2)
accuracy_for_test = np.round((DT_cm_test[0][0] + DT_cm_test[1][1]) /len(y_test) ,2)
print('Accuracy for training set for Decision Tree = {} '.format(accuracy_for_train))
print('Accuracy for test set for Decision Tree = {} '.format(accuracy_for_test))

RF_model = RandomForestClassifier(criterion='gini', max_depth=10, min_samples_split=2, n_estimators = 10, random_state=42)
RF_model.fit(X_train, y_train)
RF_y_pred = RF_model.predict(X_test)
RF_cm_train = confusion_matrix(y_train, RF_model.predict(X_train))
RF_cm_test = confusion_matrix(y_test, RF_y_pred)
accuracy_for_train = np.round((RF_cm_train[0][0] + RF_cm_train[1][1]) /len(y_train) ,2)
accuracy_for_test = np.round((RF_cm_test[0][0] + RF_cm_test[1][1]) /len(y_test) ,2)
print('Accuracy for training set for Random Forest = {} '.format(accuracy_for_train))
print('Accuracy for test set for Random Forest = {} '.format(accuracy_for_test))

AB_model = AdaBoostClassifier(n_estimators=50, learning_rate=1.0, random_state=42)
AB_model.fit(X_train, y_train)
AB_y_pred = AB_model.predict(X_test)
AB_cm_train = confusion_matrix(y_train, AB_model.predict(X_train))
AB_cm_test = confusion_matrix(y_test, AB_y_pred)
accuracy_for_train = np.round((AB_cm_train[0][0] + AB_cm_train[1][1]) /len(y_train) ,2)
accuracy_for_test = np.round((AB_cm_test[0][0] + AB_cm_test[1][1]) /len(y_test) ,2)
print('Accuracy for training set for AdaBoost = {} '.format(accuracy_for_train))
print('Accuracy for test set for AdaBoost = {} '.format(accuracy_for_test))

GB_model = GradientBoostingClassifier(learning_rate=0.1, n_estimators=100, subsample=1.0, min_samples_split=2, max_depth=3,random_state=42)
GB_model.fit(X_train, y_train)
GB_y_pred = GB_model.predict(X_test)
GB_cm_train = confusion_matrix(y_train, GB_model.predict(X_train))
GB_cm_test = confusion_matrix(y_test, GB_y_pred)
accuracy_for_train = np.round((GB_cm_train[0][0] + GB_cm_train[1][1]) /len(y_train) ,2)
accuracy_for_test = np.round((GB_cm_test[0][0] + GB_cm_test[1][1]) /len(y_test) ,2)
print('Accuracy for training set for Gradient Boosting = {} '.format(accuracy_for_train))
print('Accuracy for test set for Gradient Boosting = {} '.format(accuracy_for_test))

XGB_model = XGBClassifier(objective="binary:logistic", random_state=42, n_estimators = 100)
XGB_model.fit(X_train, y_train)
XGB_y_pred = XGB_model.predict(X_test)
XGB_cm_train = confusion_matrix(y_train, XGB_model.predict(X_train))
XGB_cm_test = confusion_matrix(y_test, XGB_y_pred)
accuracy_for_train = np.round((XGB_cm_train[0][0] + XGB_cm_train[1][1]) /len(y_train) ,2)
accuracy_for_test = np.round((XGB_cm_test[0][0] + XGB_cm_test[1][1]) /len(y_test) ,2)
print('Accuracy for training set for XGBoost = {} '.format(accuracy_for_train))
print('Accuracy for test set for XGBoost = {} '.format(accuracy_for_test))


dtc = DecisionTreeClassifier(random_state =42)
rfc = RandomForestClassifier(random_state =42)
knn = KNeighborsClassifier ()
xgb = XGBClassifier(random_state =42)
gc = GradientBoostingClassifier ( random_state =42)
svc = SVC(kernel ='rbf', random_state =42)
ad = AdaBoostClassifier ( random_state =42)



Stacking_model = StackingClassifier(estimators=[('dtc', dtc), ('rfc', rfc), ('knn', knn), ('gc', gc), ('ad', ad),('svc', svc)], final_estimator= xgb)
Stacking_model.fit(X_train, y_train)
Stacking_y_pred = Stacking_model.predict(X_test)
Stacking_cm_train = confusion_matrix(y_train, Stacking_model.predict(X_train))
Stacking_cm_test = confusion_matrix(y_test, Stacking_y_pred)
accuracy_for_train = np.round((Stacking_cm_train[0][0] + Stacking_cm_train[1][1]) /len(y_train) ,2)
accuracy_for_test = np.round((Stacking_cm_test[0][0] + Stacking_cm_test[1][1]) /len(y_test) ,2)
print('Accuracy for training set for Stacking = {} '.format(accuracy_for_train))
print('Accuracy for test set for Stacking = {} '.format(accuracy_for_test))
