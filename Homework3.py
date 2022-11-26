import numpy as np
import pandas as pd

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print('Train dataset：', train.shape, 'Test dataset：', test.shape)

print(train.isnull().sum(axis=0))

print(test.isnull().sum(axis=0))

full = train.append(test, ignore_index = True, sort = True)
print('Full dataset：', full.shape)

full.head()

full.describe()

full.info()

full['Age'] = full['Age'].fillna(full['Age'].mean())

full.drop(labels='Cabin',axis=1,inplace=True)

embarkedCount = full.groupby('Embarked')['Embarked'].count()
embarkedOrderedCount = embarkedCount.sort_values(ascending=False)
print('Embarked Number:\n',embarkedOrderedCount)

full['Embarked'] = full['Embarked'].fillna('S')

full['Fare'] = full['Fare'].fillna(full['Fare'].mean())

sexCount = full.groupby('Sex')['Sex'].count()
print('sex number：\n',sexCount)

sex_mapDict = {'male':1,'female':0}
full['Sex'] = full['Sex'].map(sex_mapDict)
full.head()

embarkedCount = full.groupby('Embarked')['Embarked'].count()
print('Embarked Number：\n', embarkedCount)

embarkedDF = pd.DataFrame()
embarkedDF = pd.get_dummies(full['Embarked'], prefix = 'Embarked')
embarkedDF.head()

full = pd.concat([full,embarkedDF], axis = 1)
full.drop('Embarked', axis = 1, inplace = True)
full.head()

pclassCount = full.groupby('Pclass')['Pclass'].count()
print('Pclass Number：\n', pclassCount)

pclassDF = pd.DataFrame()
pclassDF = pd.get_dummies(full['Pclass'], prefix = 'Pclass')
pclassDF.head()

full = pd.concat([full,pclassDF], axis = 1)
full.drop('Pclass', axis = 1, inplace = True)
full.head()

def gettitle(name):
    str1=name.split(',')[1]
    str2=str1.split('.')[0]
    str3=str2.strip()
    return str3

titleDF=pd.DataFrame()
titleDF['Title'] = full['Name'].map(gettitle)

titleCount = titleDF.groupby('Title')['Title'].count()
print('title：\n',titleCount)

title_mapDict={'Capt':'Officer',
               'Col':'Officer',
               'Major':'Officer',
               'Jonkheer':'Royalty',
               'Don':'Royalty',
               'Sir':'Royalty',
               'Dr':'Officer',
               'Rev':'Officer',
               'the Countess':'Royalty',
               'Dona':'Royalty',
               'Mme':'Mrs',
               'Mlle':'Miss',
               'Ms':'Mrs',
               'Mr':'Mr',
               'Mrs':'Mrs',
               'Miss':'Miss',
               'Master':'Master',
               'Lady':'Royalty'}

titleDF['Title'] = titleDF['Title'].map(title_mapDict)
titleDF = pd.get_dummies(titleDF['Title'], prefix = 'Title')
titleDF.head()

full = pd.concat([full,titleDF], axis = 1)
full.drop('Name', axis=1, inplace=True)
full.head()

familyDF = pd.DataFrame()
familyDF['FamilySize'] = full['Parch']+full['SibSp']+1

familyDF['Family_Single'] = familyDF['FamilySize'].map(lambda f:1 if f == 1 else 0)
familyDF['Family_Small'] = familyDF['FamilySize'].map(lambda f:1 if 2<= f <=4 else 0)
familyDF['Family_Large'] = familyDF['FamilySize'].map(lambda f:1 if f >= 5 else 0)

full = pd.concat([full,familyDF],axis=1)
full.head()

corrDF = full.corr() 
corrDF['Survived'].sort_values(ascending=False)

full_X = pd.concat([titleDF, pclassDF, familyDF, full['Fare'], embarkedDF, full['Sex']], axis = 1)
full_X.head()

sourceRow = 891
source_X = full_X.loc[0:sourceRow-1,:]
source_y = full.loc[0:sourceRow-1,'Survived']   
pred_X = full_X.loc[sourceRow:,:]
print('original row：', source_X.shape[0])
print('original row：', pred_X.shape[0])
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(source_X, source_y, train_size=0.8, test_size=0.2)
print('\n original feature：', source_X.shape,
   '\n train feature：', X_train.shape,
   '\n test feature：', X_test.shape)
print('\n original label：', source_y.shape,
   '\n train label：', y_train.shape,
   '\n test label：', y_test.shape)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver = 'liblinear')

model.fit(X_train,y_train)

fscore = model.score(X_test,y_test)
print('accuracy:',fscore)

pred_Y = model.predict(pred_X)
pred_Y = pred_Y.astype(int)

passenger_id = full.loc[sourceRow:,'PassengerId']
predDF = pd.DataFrame({'PassengerId':passenger_id,'Survived':pred_Y})
print(predDF)