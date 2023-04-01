# using panda dataframe
import pandas as pd
import numpy as np
file=pd.read_csv('/content/heart disease classification dataset.csv')
#file.head()
file=pd.DataFrame(file)
file.isnull().sum()


#input mising values
from sklearn.impute import SimpleImputer

impute = SimpleImputer(missing_values=np.nan, strategy='mean')

impute.fit(file[['trestbps']])

file['trestbps'] = impute.transform(file[['trestbps']])
impute.fit(file[['chol']])

file['chol'] = impute.transform(file[['chol']])
impute.fit(file[['thal']])

file['thalach'] = impute.transform(file[['thalach']])
file.isnull().sum()


file['target'].unique()


#encoding
from sklearn.preprocessing import LabelEncoder

# Set up the LabelEncoder object
enc = LabelEncoder()

file['target'] = enc.fit_transform(file['target'])

# Compare the two columns
print(file[['target']].head())
file

#split
from sklearn.model_selection import train_test_split

cancer=file

x=cancer.iloc[:,3:10]
y=cancer.iloc[:,11:14]

X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=1)
print(X_train.shape)
print(X_test.shape)
x
y

#scaling
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)

print("per-feature minimum before scaling:\n {}".format(X_train.min(axis=0)))
print("per-feature maximum before scaling:\n {}".format(X_train.max(axis=0)))

print("per-feature minimum after scaling:\n {}".format(
    X_train_scaled.min(axis=0)))
print("per-feature maximum after scaling:\n {}".format(
    X_train_scaled.max(axis=0)))

