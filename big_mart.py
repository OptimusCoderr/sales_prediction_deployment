"""
@author: Abdulmalik Adeyemo
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder


import pickle
import streamlit as st
from streamlit_option_menu import option_menu


# loading the saved models
model = pickle.load(open('big_mart_model.pkl', 'rb'))


# sidebar for navigation
with st.sidebar:
    
    selected = option_menu('Sales Prediction System', #Title of the Sidebar
                          
                          ['Big Mart Sales Prediction'], #You can add more options to the sidebar
                          icons=['shop'], #BootStrap Icons - Add more depending on the number of sidebar options you have.
                          default_index=0) #Default side bar selection
    
    
# Sales Prediction Page
if (selected == 'Big Mart Sales Prediction'):
    
    # page title
    st.title('Sales Prediction using ML')
    
    
    # getting the input data from the user
    col1, col2 = st.columns(2)
    
    with col1:
        Item_Visibility = st.number_input('Item Visibility', min_value=0.00, max_value=0.50, step=0.01)

    with col1:
        Item_MRP = st.number_input('Item MRP', min_value=30.00, max_value=300.00, step=1.00)

    with col1:
        Outlet_Size = st.selectbox('Outlet Size', ['Small', 'Medium', 'High'])

    with col2:
        Item_Fat_Content = st.selectbox('Item Fat Content', ['Low Fat', 'Regular'])

    with col2:
        Outlet_Location_Type = st.selectbox('Outlet Location Type', ['Tier 1', 'Tier 2', 'Tier 3'])

    data = {
            'Item_Visibility': Item_Visibility,
            'Item_MRP' : Item_MRP,
            'Outlet_Size' : Outlet_Size,
            'Item_Fat_Content': Item_Fat_Content,
            'Outlet_Location_Type' : Outlet_Location_Type
             }

    df = pd.DataFrame(data, index=[0])

    
    # code for Prediction
    sales_prediction_output = ""
    
    # creating a button for Prediction
    
    if st.button('Predict Sales'):
        sales_prediction = model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        sales_prediction_output = f"The sales is predicted to be {sales_prediction}"

        
    st.success(sales_prediction_output)





    



