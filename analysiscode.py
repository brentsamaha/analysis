#First import packages to use later on 
#Importing matplotlib.pyplot as plt, plt is a pseudonym for efficiency 

import pandas
from pandas import DataFrame
import matplotlib.pyplot as plt
import seaborn
import statsmodels.api as sm

#opens toyota data file and saves it as variable
data1=pandas.read_csv('ToyotaCorolla.csv')

#call on variable to view all columns in file
data1.columns

#assign a new variable as a data frame, using the info from our variable "data1" which includes the file
#specify which columns we want 
data=DataFrame(data1, columns=['Price','Fuel_Type','Mfg_Year','HP','KM','Color','Automatic','Doors','ABS','Parking_Assistant'])

#ensure that your dataframe has the columns you want and visualize it within a JNB
data


#get quick information about your columns, check under the non-null count to see if any values are missing per column
data.info()

#under info, some of the columns are labeled as int type, but are category
#switch the Dtype by assigning the column to category type
data['Fuel_Type']=data['Fuel_Type'].astype('category')
data['Color']=data['Color'].astype('category')
data['ABS']=data['ABS'].astype('category')
data['Parking_Assistant']=data['Parking_Assistant'].astype('category')

#use the describe feature for quick stats of your data frame
#only the int type columns will be shown
data.describe()

#for category type, specify the column to describe
data['Color'].describe()

#to get count info for an int type column
#use sort=False to have number of doors in ascending order
#not using sort=False will produce a most frequent to least frequent answer
data['Doors'].value_counts(sort=False)

#create a bar graph to visualize the same data
seaborn.countplot(x='Doors', data=data)
plt.xlabel("Doors on Vehicle")
plt.ylabel("Count")
plt.title("Frequency of # of Doors on Vehicle")

#for the automatic column, it is a category type with an answer of 0 or 1
#change type to category and rename categories so 0 and 1 are logical
data['Automatic']=data['Automatic'].astype('category')
data['Automatic']=data['Automatic'].cat.rename_categories(["Automatic", "Manual"])

#create bar graph displaying automatic types of data
seaborn.countplot(x='Automatic', data=data)
plt.xlabel("Transmission Type")
plt.ylabel("Count")
plt.title("Transmission Type Frequency")

#view any correlations between int type columns
data.corr()

#data shows a moderately strong negative correlation between price and kilometers driven on car
#visualize the correlation with a regression plot
seaborn.regplot(x='KM', y='Price', data=data)
plt.xlabel("Kilometers driven on car")
plt.ylabel('Car Price')
plt.title("Relationship between kilometers driven and vehicle price")

#view ordinary least square data for more stats of two variables
result=sm.OLS(data['Price'], data['KM']).fit()
result.summary()
