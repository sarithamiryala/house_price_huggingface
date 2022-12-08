import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import seaborn as sns



# loading the data
df = pd.read_csv('housing.csv')


# Renaming columns
df.rename(columns = {'Avg. Area Income':'Income','Avg. Area House Age':'House_age', 'Avg. Area Number of Rooms':'No_rooms',
       'Avg. Area Number of Bedrooms':'No_bedrooms', 'Area Population':'population'},inplace = True)


# HEADINGS
st.title('House Price Prediction')
st.sidebar.header('Housing Data')
st.subheader('Training Data Stats')
st.write(df.describe())


# X AND Y DATA
x = df.drop(['Price'], axis = 1)
y = df.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)


# FUNCTION
def user_report():
  Income = st.sidebar.slider('Income', 17795,107702, 18000 )
  House_age = st.sidebar.slider('House_age', 2,10, 4 )
  No_rooms = st.sidebar.slider('No_rooms', 3,11, 5 )
  No_bedrooms = st.sidebar.slider('No_bedrooms', 2,7, 3 )
  population = st.sidebar.slider('population', 170,70000, 5000 )
  

  user_report_data = {
      'Income':Income,
      'House_age':House_age,
      'No_rooms':No_rooms,
      'No_bedrooms':No_bedrooms,
      'population':population
  }
  report_data = pd.DataFrame(user_report_data, index=[0])
  return report_data




# Housing Data
user_data = user_report()
st.subheader('Housing Data')
st.write(user_data)




# MODEL
lr  = LinearRegression()
lr.fit(x_train, y_train)
user_result = lr.predict(user_data)



# VISUALISATIONS
st.title('Visualised Housing Data')



# COLOR FUNCTION
if user_result[0]==0:
  color = 'blue'
else:
  color = 'red'



# OUTPUT
st.subheader('Price of House is : ')
st.write(str(user_result))
st.title('output')
st.subheader('r2_score: ')
st.write(str(r2_score(y_test, lr.predict(x_test))*100)+'%')
