# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 15:43:43 2021

@author: Ali Kalbaliyev
"""

import pandas as pd
import streamlit as st
import plotly.express as px
from PIL import Image

st.set_page_config(page_title='Machine Learning Application')
st.header('Welcome to ML Application')


#### -- Load DataFrame 
df=pd.read_csv(r'C:/Users/dell/Desktop/Python/week8/Salary.csv')
df=df.dropna()
st.dataframe(df) # show Dataframe on app


# Creating beta columns for splitting width n columns
column1, column2 = st.columns(2)


# Piechart 
chart1=px.pie(df,
              title='Pie Chart',
              names='Country',
              values='change_company_a_lot',)
#Scatter Plot
chart2 =px.scatter(df,
                       
                   x='Salary_usd',
                   y='Experience_year',
                   color='Country',
                   title='Scatter Plot')

#Funnel Chart
# Calculation for Funnel chart
salary_mean=df.groupby("Country").agg({'Salary_usd':'mean'})
salary_mean=salary_mean.sort_values(by=['Salary_usd'], ascending=False)
salary_mean=salary_mean['Salary_usd'].reset_index()

chart3=px.funnel(salary_mean,
                 title='Funnel chart',
                 y='Country',
                 x='Salary_usd')


# Plotting on Application
column1.plotly_chart(chart1,use_container_width=True)
column2.plotly_chart(chart3,use_container_width=True)
st.plotly_chart(chart2)


## Adding Images

st.header('First Image')
image1=Image.open('meme2.jpg') # reading image 
st.image(image1,caption='First Image',use_column_width=True) # showing on application 


#Slider
unique_country=list(df["Country"].unique())
unique_ex_years=list(df["Salary_usd"].unique())

st.header('Salary Slider')
salary = st.slider(label='Select Salary',
                   min_value=float(min(unique_ex_years)),
                   max_value=float(max(unique_ex_years)),
                   value=float(min(unique_ex_years)))

st.write("Selected", salary, 'USD') # real-time checking slider value


#Multi Selection
country_selection=st.multiselect('Countries Select Menu' ,
                                 unique_country,unique_country)


#Filter DataFrame
df_filter=(df["Salary_usd"]>salary) &(df["Country"].isin(country_selection)) # condition

result_number=df[df_filter].shape[0] # results
st.markdown('Number of result: {0}'.format(result_number)) # realtime showing result number
st.dataframe(df[df_filter])




# Barchart
# Calculation for Bar chart
salary_count=df[df_filter].groupby("Country").agg({'Salary_usd':'size'})
salary_count=salary_count.sort_values(by=['Salary_usd'], ascending=False)
salary_count=salary_count['Salary_usd'].reset_index()

barchart=px.bar(df[df_filter], 
                title='Bar Chart',
                x=salary_count["Country"],
                y=salary_count["Salary_usd"],
                template='plotly_white',
                color_discrete_sequence=['#800080'])

st.plotly_chart(barchart)


#Modelling
st.header('Modelling with Logistic Regression')
st.subheader('Diabetes Dataset')

diabetes=pd.read_csv(r"C:\Users\dell\Desktop\Python\week8\diabetes.csv")
st.dataframe(diabetes)

# Splitting 
X=diabetes.drop("Outcome",axis=1)
Y=diabetes.Outcome

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=42)

st.markdown('X_train size = {0}'.format(X_train.shape))
st.markdown('X_test size = {0}'.format(X_test.shape))
st.markdown('y_train size = {0}'.format(y_train.shape))
st.markdown('y_test size = {0}'.format(y_test.shape))

if st.button('Calculate Model'): # add Button to Application
    st.title('Congratulations Your Model is working')
    
    import pickle
    document="myModel"
    loaded_model=pickle.load(open(document,'rb'))
    y_loded_model_pred=loaded_model.predict(X_test)
    
    from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
    st.markdown('Confusion Matrix')
    st.write(confusion_matrix(y_test,y_loded_model_pred))
    
    report = classification_report(y_test, y_loded_model_pred, output_dict=True) # creating dataframe from classifaction report
    df_report = pd.DataFrame(report).transpose()
    
    st.dataframe(df_report)
    
    accuracy=str(round(accuracy_score(y_test,y_loded_model_pred),2))+"%"
    st.markdown("Accuracy Score = "+accuracy)   
    st.title('Thanks For using')




