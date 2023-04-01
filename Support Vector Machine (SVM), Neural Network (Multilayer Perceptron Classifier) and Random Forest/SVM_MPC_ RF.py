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
# cancer=file
x=file.iloc[:,1:17]
# y=cancer.iloc[:,17]
y=file['class_type']

X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=1)
print(X_train.shape)
print(X_test.shape)
file

#Support Vector Machine
from sklearn.svm import SVC
svc = SVC(kernel="linear")
# svc = SVC(kernel="sigmoid")
svc.fit(X_train, y_train)
y_prsd1=svc.predict(X_test)
before_SVM=accuracy_score(y_test,y_prsd1)
print(before_SVM)

#Performance without dimension reduction
print("Training accuracy is {:.2f}".format(svc.score(X_train, y_train)) )
print("Testing accuracy is {:.2f} ".format(svc.score(X_test, y_test)) )

#Neural Network (MLPClassifier)
from sklearn.neural_network import MLPClassifier
nnc=MLPClassifier(hidden_layer_sizes=(8), activation="relu", max_iter=10000)
# nnc=MLPClassifier(hidden_layer_sizes=(4),max_iter=10000)
nnc.fit(X_train, y_train)
predictions1_2 = nnc.predict(X_test)
before_nn=accuracy_score(y_test,predictions1_2)
print(before_nn)

#Performance without dimension reduction
print("Training accuracy is {:.2f}".format(nnc.score(X_train, y_train)) )
print("Testing accuracy is {:.2f} ".format(nnc.score(X_test, y_test)) )

#Random Forest
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100)
# rfc = RandomForestClassifier(max_depth=4)
rfc.fit(X_train, y_train)
predictions = rfc.predict(X_test)
before_rf=accuracy_score(y_test,predictions)
print(before_rf)

#Performance without dimension reduction
print("Training accuracy is {:.2f}".format(rfc.score(X_train, y_train)) )
print("Testing accuracy is {:.2f} ".format(rfc.score(X_test, y_test)) )

#PCA
from sklearn.decomposition import PCA
n=len(x.columns)
print((n/2)) 
pca = PCA(n_components=int(n/2))

principal_components= pca.fit_transform(X_test)
print(principal_components)

n=int((n/2)+1)
cols=["principle component "+str(i) for i in range(1,n)]
print(cols)

principal_df = pd.DataFrame(data=principal_components,columns=cols)
main_df=pd.concat([principal_df, file[["class_type"]]], axis=1)

main_df.head()

main_df.isnull().sum()
#input missing values
from sklearn.impute import SimpleImputer

impute = SimpleImputer(missing_values=np.nan, strategy='median')

impute.fit(main_df[['principle component 1']])

main_df['principle component 1'] = impute.transform(main_df[['principle component 1']])
impute.fit(main_df[['principle component 2']])
main_df['principle component 2'] = impute.transform(main_df[['principle component 2']])
impute.fit(main_df[['principle component 3']])
main_df['principle component 3'] = impute.transform(main_df[['principle component 3']])
impute.fit(main_df[['principle component 4']])
main_df['principle component 4'] = impute.transform(main_df[['principle component 4']])
impute.fit(main_df[['principle component 5']])
main_df['principle component 5'] = impute.transform(main_df[['principle component 5']])
impute.fit(main_df[['principle component 6']])
main_df['principle component 6'] = impute.transform(main_df[['principle component 6']])

impute.fit(main_df[['principle component 7']])
main_df['principle component 7'] = impute.transform(main_df[['principle component 7']])

impute.fit(main_df[['principle component 8']])
main_df['principle component 8'] = impute.transform(main_df[['principle component 8']])

main_df.isnull().sum()

main_df.head()

#split
from sklearn.model_selection import train_test_split

x=main_df.iloc[:,0:8]
# y=main_df.iloc[:,8]
y=main_df['class_type']

X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=1)
print(X_train.shape)
print(X_test.shape)

#Support Vector Machine
from sklearn.svm import SVC
svc = SVC(kernel="linear")
# svc = SVC(kernel="sigmoid")
svc.fit(X_train, y_train)
y_prsd2=svc.predict(X_test)
after_SVM=accuracy_score(y_test,y_prsd2)
print(after_SVM)

#Performance with dimension reduction
print("Training accuracy is {:.2f}".format(svc.score(X_train, y_train)) )
print("Testing accuracy is {:.2f} ".format(svc.score(X_test, y_test)) )

#Neural Network (MLPClassifier) 
from sklearn.neural_network import MLPClassifier
# nnc=MLPClassifier(hidden_layer_sizes=(7), activation="relu", max_iter=10000)
nnc=MLPClassifier(hidden_layer_sizes=(8),max_iter=10000)
nnc.fit(X_train, y_train)
predictions1 = nnc.predict(X_test)
after_nn=accuracy_score(y_test,predictions1)
print(after_nn)

#Performance with dimension reduction
print("Training accuracy is {:.2f}".format(nnc.score(X_train, y_train)) )
print("Testing accuracy is {:.2f} ".format(nnc.score(X_test, y_test)) )

#Random Forest
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100)
# rfc = RandomForestClassifier(max_depth=2)
# rfc = RandomForestClassifier(max_depth=8)
rfc.fit(X_train, y_train)
predictions2 = rfc.predict(X_test)
after_rf=accuracy_score(y_test,predictions2)
print(after_rf)

#Performance with dimension reduction
print("Training accuracy is {:.2f}".format(rfc.score(X_train, y_train)) )
print("Testing accuracy is {:.2f} ".format(rfc.score(X_test, y_test)) )

#Bar chart Plotting
from sklearn import tree
import matplotlib.pyplot as plt
labels = ['Support Vector Machine', 'Neural Network', 'Random Forest']

before=[before_SVM,before_nn,before_rf]
after=[after_SVM,after_nn,after_rf]
x = np.arange(len(labels)) 
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, before, width, label='Pre-PCA')
rects2 = ax.bar(x + width/2, after, width, label='Post-PCA')

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

fig.tight_layout()

plt.show()






