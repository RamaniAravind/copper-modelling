# Importing Libraries
import pickle
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

# Setting Webpage Configurations

st.set_page_config(
    page_icon="⚙",  
    page_title="Copper modeling",  
    layout="wide"  
)

st.markdown(
    "<h1 style='color: brown; text-align: center; text-transform: uppercase;'>COPPER MODELING</h1>",
    unsafe_allow_html=True
)


# Header Image

image = Image.open(r'C:\Users\Acer\Desktop\copper\copper.png')

# Define desired width and height
desired_width = 200
desired_height = 100

# Resize the image
resized_image = image.resize((desired_width, desired_height))

# Calculate the padding required for center alignment
padding = (st.sidebar.empty(), st.empty(), st.sidebar.empty())

# Display the resized image
st.image(resized_image, caption='Copper Model', use_column_width=False)

# Align the image in the center
for pad in padding:
    pad.markdown("<div style='text-align: center;'> </div>", unsafe_allow_html=True)

# Loading the model

def load_model():
    model = pickle.load(open('price_prediction.pkl','rb'))
    return model


def load_encoder():
    encoder = pickle.load(open('encoder.pkl','rb'))
    return encoder


def load_class_model():
    model = pickle.load(open('Status_prediction.pkl','rb'))
    return model

# Price prediction model
reg_model = load_model()

# encoder
encoder = load_encoder()

# Status Prediction model
class_model = load_class_model()

# Reading the Dataframe
model_df = pd.read_csv(r'Copper.csv')

# Querying Win/Lost status
query_df = model_df.query("status == 'Won' or status == 'Lost'")

tab1,tab2 = st.tabs(['Selling Price Prediction','Status Prediction'])

with tab1:

    item_year = st.selectbox('Select the Item year',options = query_df['item_date'].value_counts().index.sort_values())

    country = st.selectbox('Select the Country Code',options = query_df['country'].value_counts().index.sort_values())

    item_type = st.selectbox('Select the Item type',options = query_df['item type'].unique())

    application = st.selectbox('Select the Application number',options = query_df['application'].value_counts().index.sort_values())

    product_ref = st.selectbox('Select the Product Category',options = query_df['product_ref'].value_counts().index.sort_values())

    delivery_year = st.selectbox('Select the Delivery year',options = query_df['delivery date'].value_counts().index.sort_values())

    thickness = st.number_input('Enter the Thickness')
    log_thickness = np.log(thickness)

    width = st.number_input('Enter the width')
    log_width = np.log(width)

    quantity_tons = st.number_input('Enter the Quantity (in tons)')
    log_quantity = np.log(quantity_tons)

    submit = st.button('Predict Price')

    if submit:
    
        user_input = pd.DataFrame([[item_year,country,item_type,application,log_thickness,log_width,product_ref,delivery_year,quantity_tons]],
                            columns = ['item_date','country','item type','application','thickness','width','product_ref','delivery date','quantity_tons'])
        
       
        prediction = reg_model.predict(user_input)
        
        selling_price = np.exp(prediction)
        
        st.subheader(f':green[Predicted Price] : {round(selling_price[0])}')

with tab2:

    country = st.selectbox('Select any one Country Code',options = query_df['country'].value_counts().index.sort_values())

    item_type = st.selectbox('Select any one Item type',options = query_df['item type'].unique())

    product_ref = st.selectbox('Select any one Product Category',options = query_df['product_ref'].value_counts().index.sort_values())

    delivery_year = st.selectbox('Select a Delivery year',options = query_df['delivery date'].value_counts().index.sort_values())

    thickness = st.number_input('Enter an Thickness')
    log_thickness = np.log(thickness)

    width = st.number_input('Enter an width')
    log_width = np.log(width)

    selling_price = st.number_input('Enter an Selling Price')
    log_selling_price = np.log(selling_price)

    quantity_tons = st.number_input('Enter an Quantity (in tons)')
    log_quantity = np.log(quantity_tons)

    user_input_1 = pd.DataFrame([[country,item_type,log_thickness,log_width,product_ref,delivery_year,log_selling_price,log_quantity]],
                       columns = ['country','item type','thickness','width','product_ref','delivery date','selling_price','quantity_tons'])
    
    submit1 = st.button('Predict')

    if submit1:
        transformed_data = encoder.transform(user_input_1)
        prediction = class_model.predict(transformed_data)
        
        if prediction[0] == 1:
            st.subheader(':green[Predicted Status] : Won')
        else:
            st.subheader('green[Predicted Status] : Lost')
