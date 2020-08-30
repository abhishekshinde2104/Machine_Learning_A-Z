import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Importing dataset
dataset=pd.read_csv('Ads_CTR_Optimisation.csv')

#Implementing UCB
import math
N=10000
d=10
ads_selected=[]#vector of each ad selected in 10k rounds
numbers_of_selections=[0]*d#no of times ad i was selected
sums_of_rewards=[0]*d
total_reward=0
for n in range(0,N):
    ad=0
    max_upper_bound=0
    for i in range(0,d):
        if(numbers_of_selections[i]>0):
            #if ad version i is selected at least once we will select the strategy
            average_reward=sums_of_rewards[i]/numbers_of_selections[i]
            #sums of rewards of ad i up to round n / number of times ad i was selected up to round n
            delta_i=math.sqrt(3/2 * math.log(n+1)/numbers_of_selections[i])#since n starts at 0 so take n+1
            upper_bound=average_reward + delta_i
            
        else:
            #here we select 1st 10 ads without startegy and then at round 11 we will have some info and sum_of_rewards
            #this else selects 1st 10 ads as per each round due to the fact the upperbound is set large
            """for n=0 we will go through 10 versions of ad and since at 1st(i=0) round no ad was selected if(number_of_selections[i]>0) 
                will nevere be true and therefore it goes to else and sets the upper_bound=1e400.
                then max_upper_bound=upper_bound=1e400
                then we go to next step i.e i=1 i.e 2nd ad
                and this 2nd ad hasnt been selected yet and therefore upper_bound=1e400
                and then the if(upper_bound > max_upper_bound) and this if statement turns FALSE and 
                therfore ad isnot set 1 and ad remians 0 and therefore for n=0 ad=0 is selected 
                n=1 ad=1 is selected upto n=9 ad=9 is selected
                for n=1 numbers_of_selections[0]>0 and not not for numbers_of_selections[1] and therfore the same way ad 1 is selected
                """
            upper_bound=1e400
            
        if upper_bound > max_upper_bound:
            #this way we will compute the different upper bounds of each of the 10ads at that round(row)
            #at 1st this max_upper_bound=0
            #then we will compute the 1st upper bound n ofcourse it will be > 0
            #so this will happen for every 1st ad
            #and so on we calculate the upper bound for different ads and replace if its greater
            max_upper_bound=upper_bound
            ad=i
            #we keep track of the ad that has max_upper_bound
    ads_selected.append(ad)
    numbers_of_selections[ad]=numbers_of_selections[ad]+1
    reward=dataset.values[n,ad]
    sums_of_rewards[ad]=sums_of_rewards[ad]+reward
    total_reward=total_reward+reward

#Visualising the results
plt.hist(ads_selected)
plt.title('Histogram of Ads Selections')
plt.xlabel('Ads')
plt.ylabel('No.of times each add was selected')
plt.show()
# at beginning we dont have much info abt the ads 
"""we dont have much info abt whether they earned their reward or whether reward =1 or 0 
we dont know bcoz we havent selected them yet 
So thats why we need to deal with the initial conditions and choose which ads we will select
during the 10 first rounds
and therefore we will simply select the 10 1st ads with no strategy here
awe will use  the strategy as soon as we some info abt the reward of each of the 10 ads"""
