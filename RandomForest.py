import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

df_train = pd.read_csv('train.csv')

#edit train data
genders_mapping = dict([('female', 0), ('male', 1)])
df_train['Sex_Val'] = df_train['Sex'].map(genders_mapping).astype(int)

embarked_locs_mapping = dict([('C', 0),('Q', 1), ('S', 2), ('NaN', 3)])
df_train['Embarked_Val'] = df_train['Embarked'].map(embarked_locs_mapping)
df_train['Embarked_Val'] = df_train['Embarked_Val'].fillna(2)
df_train['Age'] = df_train['Age'].fillna(df_train['Age'].median())
df_train['Fare'] = df_train['Fare'].fillna(df_train['Fare'].median())
df_train['FamilySize'] = df_train['SibSp'] + df_train['Parch']
df_train = df_train.drop(['PassengerId', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'SibSp', 'Parch'], axis=1)
train_data = df_train.values

#train random forest classifier
clf = RandomForestClassifier(random_state = 10, warm_start = True, n_estimators = 26, max_depth = 6, max_features = 'sqrt')
train_features = train_data[:,1:]
train_target = train_data[:, 0]

#cross validation
train_x, test_x, train_y, test_y = train_test_split(train_features, train_target, test_size=0.20, random_state=0)
clf = clf.fit(train_x, train_y)
predict_y = clf.predict(test_x)
print ("Accuracy = %.2f" % (accuracy_score(test_y, predict_y)))

#edit test data
df_test = pd.read_csv('test.csv')
df_test['Sex_Val'] = df_test['Sex'].map(genders_mapping)
df_test['Embarked_Val'] = df_test['Embarked'].map(embarked_locs_mapping)
df_test['Family_Size'] = df_test['SibSp'] + df_test['Parch']
df_test = df_test.drop(['Name','Sex','SibSp','Parch','Ticket','Cabin','Embarked'], axis=1)
df_test['Age'] = df_test['Age'].fillna(df_test['Age'].median())
df_test['Fare'] = df_test['Fare'].fillna(df_test['Fare'].median())

#predict survival
predict_test = clf.predict(df_test.values[:,1:]).astype(int)
df_test['Survived'] = predict_test
df_test[['PassengerId', 'Survived']].to_csv('random_forest_result.csv', index=False)



