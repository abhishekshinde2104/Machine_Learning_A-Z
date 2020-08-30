#Apriori

#Importing Librabries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Importing Dataset
dataset=pd.read_csv('Market_Basket_Optimisation.csv',header=None)
#each line means new customer or same customer another day
#apyori takes a list of lists
#so there will be 1 big list containing all different transactions and each transactions
#is going to be a list itself

#Creating list of lists
transactions=[]
for i in range(0,7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0,20)])
#1st for loop for going through all the transactions 


#Training Apriori on the dataset

#takes transactions as input and rules as ouput
#transactions should be a list of lists of strings of the products
    
from apyori import apriori
rules1=apriori(transactions,min_support=0.003,min_confidence=0.2,min_lift=3,min_length=2)
rules2=apriori(transactions,min_support=0.003,min_confidence=0.2,min_lift=3,min_length=2)
#Visualising the results
results=list(rules1)
list(rules2)



















