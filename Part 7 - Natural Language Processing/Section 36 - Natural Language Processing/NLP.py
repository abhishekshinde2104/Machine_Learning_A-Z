import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Importing Dataset
dataset=pd.read_csv('Restaurant_Reviews.tsv',delimiter='\t',quoting=3)
#qutoing=3 ingnores the double quotes 

# Cleaning the texts 
#this is impt as when u create the bag of words it gets rid off the unrelevant words like a,the etc..
#this bag consists of only relevant words which help the ml to predict if the review is +ve or -ve
#we will also do stemming i.e loved will be stored as love as it gives same meaning of the --> this is
#done as in the end we dont have many with same meaning

import re #Regular Expression 
import nltk 
nltk.download('stopwords')#stopwords contains all the words that are irrelevant
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer#used for stemming
#done to remove words like loved,will loved with just love 
#as to reduce the words in the sparse matrix 
#different words corresponding to same root give the algo the same hint and meaning +,-
corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])#this removes all the punctuation marks and all other numbers and special characters
    review = review.lower()#all the letters are made to lower case
    review = review.split()#we split the string of reviews into words in that review 
    ps = PorterStemmer()#object of the class
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    #we make a for loop to go through all the words in a review and remove the words
    #that are in stopwords which are irrelevant
    #we apply stemming in to each word in the review
    #ps.stem() method is used to stem the word
    review = ' '.join(review)
    #this will join the cleaned words back to string
    corpus.append(review)
    
# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
#the cleaning of the text can be done here also with different parameter but this is not best way
cv = CountVectorizer(max_features = 1500)#max_features =1500 will keep only 1500 most frequent words 
X = cv.fit_transform(corpus).toarray()#this creates new column for each new word 
#X is the sparse matrix of features X
#we can reduce sparsity by 1)max_feature 2)Reducing Dimensionality
y = dataset.iloc[:, 1].values


"""we 1st take different words from the corpus and create 1 column for each word
and then we put the columns in a table where rows are the  1000 reviews .
table-->r * c ==> 1000 reviews * different words
each cell will contain a number and this number corresponds to each time the word 
appeared in the review and as this contains many zeros this is called sparse matrix
Tokenisation is taking all diff words of the review and creating 1 column for each of these words
WHY?? 
ML model will make some correlations between the words and the review
Each of the word in the column corresponds to 1 independent variable bcoz each of these columns
that are words are in some connected to the reviews bcoz for each of the review we can say 
if the word appears yes or no 1 or 0 the corresponding column gets
this allows us to create classification model """


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Training the Naive Bayes model on the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

#No NEED OF FEATURE SCALING

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

(55+91)/200