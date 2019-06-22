

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('train.csv')
X= dataset.iloc[:,2].values


#cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus =[]
for i  in range (0, 1000):
    #only have letters from a to z and A to Z
    review =re.sub('[^a-zA-Z]',' ', dataset['Review'][i])
    #convert to lowercase
    review =review.lower()
    #split into words from the string
    review=review.split()
    ps=PorterStemmer()
    #remove words that are in the stopwords library(it contains words that are irelevant to our model)
    #stemming which is taking the root of a word(loved to love)
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review=' ' .join(review)
    corpus.append(review)
#max features paremeters in cv removes the words which only appear once
#creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv =CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,1].values


#splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, ytest =train_test_split(X, y, test_size = 0.20, random_state = 0)

#Feature scaling
'''from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train =sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)'''

#fitting classifier to the training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

#test
y_pred =classifier.predict(X_test)

#Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm =confusion_matrix(ytest, y_pred)