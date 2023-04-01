# using panda dataframe
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
file=pd.read_csv('/content/zoo.csv')
#file.head()
file=pd.DataFrame(file)
file.isnull().sum()


#split
from sklearn.model_selection import train_test_split

X=file.iloc[:,1:17]
y=file['class_type']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=1)
print(X_train.shape)
print(X_test.shape)


#LogisticRegression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=500)
model.fit(X_train,y_train)
y_pred1=model.predict(X_test)
score1=accuracy_score(y_test,y_pred1)
print(score1)

model.predict([[1,0,1,0,0,1,1,1,1,0,0,1,0,1,0,0]])

#DecisionTree
from sklearn.tree import DecisionTreeClassifier

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)
clf = DecisionTreeClassifier(criterion='entropy',random_state=1)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
score2=accuracy_score(y_test,y_pred)
print(score2)

clf.predict([[1,0,1,0,0,1,1,1,1,0,0,1,0,1,0,0]])

# plot bar chart
import matplotlib.pyplot as plt
plt.bar(['LogisticRegression','DecisionTree'], [score1,score2])
plt.show()


