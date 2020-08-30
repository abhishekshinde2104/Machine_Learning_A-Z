import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Importing dataset
dataset=pd.read_csv('Ads_CTR_Optimisation.csv')

#Implementing Thompson Sampling
import random
N=10000
d=10
ads_selected=[]
numbers_of_rewards_1=[0]*d
numbers_of_rewards_0=[0]*d
total_reward=0
for n in range(0,N):
    ad=0
    max_random=0
    for i in range(0,d):
        random_beta=random.betavariate(numbers_of_rewards_1[i]+1,numbers_of_rewards_0[i]+1)#This gives the random draws
        #for each ad i in this for i loop here
        #we are taking a random draw from the random_beta
        #each time we take a random draw we check to see if this random draw(random_beta) is higher than max_random
        #this condition will be true for 1st ad
        
        #after that it selects the ad that max_random_draw and forget about previous random draw ad  as it has lowest random draw
        
        
        if random_beta > max_random:
            max_random=random_beta
            ad=i
            
    ads_selected.append(ad)
    reward=dataset.values[n,ad]
    if reward==1:
        numbers_of_rewards_1[ad]=numbers_of_rewards_1[ad]+1
    else:
        numbers_of_rewards_0[ad]=numbers_of_rewards_0[ad]+1
    #numbers_of_rewards_1 --> different numbers of times each ad got reward 1 at round n
    #numbers_of_rewards_0 -->different numbers of times each ad got reward 0 at round n
    total_reward=total_reward+reward

#Visualising the results
plt.hist(ads_selected)
plt.title('Histogram of Ads Selections')
plt.xlabel('Ads')
plt.ylabel('No.of times each add was selected')
plt.show()