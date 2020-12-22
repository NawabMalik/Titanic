#1-Collecting the data.......
import pandas as pd
train=pd.read_csv("D:\\Study\\Python\\scripts\\Machine_Learning\\Supervised_Model\\Logistic_Regression\\Titanic\\train.csv")

train.head()
train.columns
train.shape
len(train)

#Analyze the data/explore as much you can........
train.isnull().sum()
train.describe()
train.info()

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

sb.countplot(x='Survived', data=train)
sb.countplot(x='Survived', hue='Sex', data=train) # hue=catagorical
sb.countplot(x='Survived', hue='Pclass', data=train) # hue=catagorical
train['Age'].plot.hist()
train['Fare'].plot.hist(bins=20, figsize=(10,5))
sb.countplot(x='SibSp', data=train)
sb.countplot(x='Parch', data=train)
sb.countplot(x='Embarked', data=train)
sb.countplot(x='Survived', hue='Embarked', data=train)

# EDA/data wrangling...........
train.isnull().sum()
train.Age.replace(to_replace=np.nan, value=train.Age.mean(),inplace=True)
train.drop('Cabin', axis=1, inplace=True)
train.columns

#Remove na from Embarked
sb.heatmap(train.isnull(),yticklabels=False, cmap='viridis')
train[train.Embarked.isnull()]
train[train.Fare==80]
sb.boxplot(x='Embarked', y='Fare', data=train)
train.Embarked.replace(to_replace=np.nan, value='C', inplace=True)
train['Embarked'].value_counts()
train['Embarked'].unique()
train.isnull().sum()

#checking duplicates
train.duplicated().sum()

train.drop(['Name','Ticket'],axis=1, inplace=True)
train1=train.drop('PassengerId',axis=1)
train1.columns

train1=pd.get_dummies(train1)
train1.head()

X=train1.drop('Survived', axis=1)
Y=train1['Survived']
print(X)
print(Y)


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y, test_size=0.2, random_state=0)
print(X_train)
print(X_test)
print(Y_train)
print(Y_test)

from sklearn.linear_model import LogisticRegression
model=LogisticRegression(C=0.1, penalty='l2', solver='lbfgs',max_iter=1000)

#Hyper_parameters tuning, for misclassification of model, penally parameter is C
# ideally C=1, we can hit/try to get best accuracy, solver only support l2 not l1
# penalty as two penalties exists, ideally C value 1.

# at c=0.1, accuracy increase to 81%
# at c=0.01, accuracy increase to 77%

print(model)                                  # predict_test, Y_train
y=model.fit(X_train,Y_train)
predicted=model.predict(X_test)

# How to set threshold/cut off on probability
"""
predicted2=model.predict_proba(X_test)
print(predicted2)
df=pd.DataFrame(predicted2, columns=['a','b'])
import numpy as np
term=np.where(df['a']>=0.8, 'YES','NO')
print(term)
term1=np.where(df['b']>=0.8, 'YES','NO')
print(term1)
"""

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
print(accuracy_score(predicted,Y_test))      #accuracy= 80%
print(confusion_matrix(predicted,Y_test))
print(classification_report(predicted,Y_test))

# check accuracy on test data.
y=model.fit(X_test,Y_test)
predicted1=model.predict(X_train)
print(accuracy_score(predicted1,Y_train))       #accuracy=77%











### other ways of feature selections
"""
#1-Collecting the data.......
import pandas as pd
train=pd.read_csv("C:\\Users\\Dell\\Desktop\\scripts\\Machine_Learning\\Supervised_Model\\Logistic_Regression\\Titanic\\train.csv")

train.head()
train.columns
train.shape
len(train)

#Analyze the data/explore as much you can........
train.isnull().sum()
train.describe()
train.info()

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

sb.countplot(x='Survived', data=train)
sb.countplot(x='Survived', hue='Sex', data=train) # hue=catagorical
sb.countplot(x='Survived', hue='Pclass', data=train) # hue=catagorical
train['Age'].plot.hist()
train['Fare'].plot.hist(bins=20, figsize=(10,5))
sb.countplot(x='SibSp', data=train)
sb.countplot(x='Parch', data=train)
sb.countplot(x='Embarked', data=train)
sb.countplot(x='Survived', hue='Embarked', data=train)

# EDA/data wrangling...........
train.isnull().sum()
train.Age.replace(to_replace=np.nan, value=train.Age.mean(),inplace=True)
train.drop('Cabin', axis=1, inplace=True)
train.columns

#Remove na from Embarked
sb.heatmap(train.isnull(),yticklabels=False, cmap='viridis')
train[train.Embarked.isnull()]
train[train.Fare==80]
sb.boxplot(x='Embarked', y='Fare', data=train)
train.Embarked.replace(to_replace=np.nan, value='C', inplace=True)
train['Embarked'].value_counts()
train['Embarked'].unique()

train.columns
train.head()
Sex=pd.get_dummies(train['Sex'],drop_first=True)
print(Sex)
Embarked=pd.get_dummies(train['Embarked'],drop_first=True)
print(Embarked)
Pclass=pd.get_dummies(train['Pclass'],drop_first=True)
print(Pclass)

train.columns
train=pd.concat([train,Sex,Embarked,Pclass], axis=1)
train.drop(['PassengerId','Sex','Pclass','Embarked'], axis=1, inplace=True)
train.columns
train.head()
X=train.drop('Survived',axis=1)
Y=train['Survived']
X.head()
Y.head()

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2, random_state=0)

from sklearn.linear_model import LogisticRegression
model=LogisticRegression(C=0.1,solver='lbfgs', max_iter=1000)
print(model)
y=model.fit(X_train,Y_train)
predicted=model.predict(X_test)

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
print(accuracy_score(predicted,Y_test))       #0.7932 / 0.81%

#accuracy on test data
y=model.fit(X_test,Y_test)
predicted1=model.predict(Y_train)
print(accuracy_score(predicted1,Y_train))       #accuracy=0.78%
"""









"""
# We can do feature selection without dummies with labelencoding.
it does 1/2/3 within class, don't create new columns

#1-Collecting the data.......
import pandas as pd
train=pd.read_csv("C:\\Users\\Dell\\Desktop\\scripts\\Machine_Learning\\Supervised_Model\\Logistic_Regression\\Titanic\\train.csv")

train.head()
train.columns
train.shape
len(train)

#Analyze the data/explore as much you can........
train.isnull().sum()
train.describe()
train.info()

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

sb.countplot(x='Survived', data=train)
sb.countplot(x='Survived', hue='Sex', data=train) # hue=catagorical
sb.countplot(x='Survived', hue='Pclass', data=train) # hue=catagorical
train['Age'].plot.hist()
train['Fare'].plot.hist(bins=20, figsize=(10,5))
sb.countplot(x='SibSp', data=train)
sb.countplot(x='Parch', data=train)
sb.countplot(x='Embarked', data=train)
sb.countplot(x='Survived', hue='Embarked', data=train)

# EDA/data wrangling...........
train.isnull().sum()
train.Age.replace(to_replace=np.nan, value=train.Age.mean(),inplace=True)
train.drop('Cabin', axis=1, inplace=True)
train.columns

#Remove na from Embarked
sb.heatmap(train.isnull(),yticklabels=False, cmap='viridis')
train[train.Embarked.isnull()]
train[train.Fare==80]
sb.boxplot(x='Embarked', y='Fare', data=train)
train.Embarked.replace(to_replace=np.nan, value='C', inplace=True)
train['Embarked'].value_counts()
train['Embarked'].unique()

train.columns
train.head()

from sklearn.preprocessing import LabelEncoder
l=LabelEncoder()
Sex=pd.DataFrame(l.fit_transform(train['Sex']))
Embarked=pd.DataFrame(l.fit_transform(train['Embarked']))
print(Sex)
print(Embarked)
type(Sex)

train=pd.concat([train,Sex,Embarked],axis=1)
train.head(25)
train.columns
train.drop(['Name','PassengerId','Ticket'],axis=1,inplace=True)
train.drop(['Embarked','Sex'],axis=1, inplace=True)
X=train.drop('Survived',axis=1)
Y=train['Survived']
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y, test_size=0.2, random_state=0)

from sklearn.linear_model import LogisticRegression
model=LogisticRegression()

y=model.fit(X_train,Y_train)
predicted2=model.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(predicted2,Y_test))           # accuracy 0.7932

"""




