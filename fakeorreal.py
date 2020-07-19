

import pandas as pd 
import numpy as np 
import pickle
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv('news.csv')
print(df.head())
print(df.isnull().sum())
print(df.columns)
print(df.describe())
print(df.info())
#df.dropna()
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

df.columns = ['newsid','title','text','label']
print(df.isnull().sum())
#Identify the number of NAs in each feature and select only those having NAs
totalna= df.isnull().sum()[df.isnull().sum() != 0]
print(totalna)

# Calculate the percentage of NA in each feature
percentageofna = totalna/df.shape[0]
print(percentageofna)
df.dropna(axis=0, inplace= True)
df.isnull().sum()

print(df.shape)
print(df.head())
#splitting training and testing set
labels = df.label
print(labels.head())
X_train, X_test, y_train, y_test = train_test_split(df['text'],labels,test_size=0.2)
print(X_train.shape,X_test.shape)
print(X_test.shape,y_test.shape)

tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

# Fit and transform train set, transform test set
tfidf_train = tfidf_vectorizer.fit_transform(X_train) 
tfidf_test  = tfidf_vectorizer.transform(X_test)

# Initialize a PassiveAggressiveClassifier
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)


y_pred=pac.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')
confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])


# saving vectorizer
with open('tfid.pickle','wb') as f:
    pickle.dump(tfidf_vectorizer,f)

# saving model
with open('model_fakenews.pickle','wb') as f:
    pickle.dump(pac,f)

