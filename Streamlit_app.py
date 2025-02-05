import streamlit as st
import pickle
import numpy as np
import pandas as pd
import base64
from sklearn.preprocessing import OrdinalEncoder


@st.cache_resource
# Load models
def load_model(model_path):
    with open(model_path, 'rb') as file:
        return pickle.load(file)

# Set background image
def set_background_image_local(image_path):
    with open(image_path, "rb") as file:
        data = file.read()
    base64_image = base64.b64encode(data).decode("utf-8")
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{base64_image}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background_image_local("car2.jpg")


# Load models and encoders
model_car=load_model("final_carmodel.pkl")
encoder_city=load_model("encoder_city.pkl")
encoder_Insurance_Validity=load_model("encoder_Insurance_Validity.pkl")
encoder_bt=load_model("encoder_bt.pkl")
encoder_ft=load_model("encoder_ft.pkl")
encoder_oem=load_model("encoder_oem.pkl")
encoder_model=load_model("encoder_model.pkl")
encoder_transmission=load_model("encoder_transmission.pkl")
encoder_variantName=load_model("encoder_variantName.pkl")

ml_df=pd.read_excel("ml_dl.xlsx")
st.title("Car Price Prediction")

categorical_features = ["city", "ft", "bt", "transmission", "oem", "model", "variantName", "Insurance Validity"]
dropdown_options = {feature: ml_df[feature].unique().tolist() for feature in categorical_features}

# Inject custom CSS to style the tabs
tab1, tab2 = st.tabs(["Home", "Price Values"])

with tab1:
    st.write("<h4 style = 'color: black;'>Welcome to the Home tab!</h4>", unsafe_allow_html=True)
    st.markdown("""
    <div style='color: black; font-size: 20px;'>
        <strong><h4 style= 'color: white;'>Introduction</h4></strong>
        In the rapidly evolving automotive market, determining the right price for a vehicle is crucial 
        for both buyers and sellers. The Car Price Prediction App provides an intelligent solution to 
        estimate car prices based on key parameters using machine learning models. This tool helps users 
        make data-driven decisions by leveraging historical data and predictive analytics.<br><br>
        
    <strong><h4 style= 'color: white;'>Key Features</h4></strong>
        <ul>
            <li>User-Friendly Interface: Simple and interactive Streamlit-based UI.</li>
            <li>Machine Learning Model: Utilizes an advanced regression model (XGBRegressor) trained 
            on a vast dataset of car prices.</li>
            <li>Feature Inputs: Users can enter details like car brand, model, manufacturing year, fuel type, 
            transmission, and other relevant attributes.</li>
            <li>Real-Time Predictions: Provides instant car price estimates based on input parameters.</li>
            <li>Comparison Tool: Allows users to compare multiple cars for better decision-making.</li>
        </ul>
        
    <strong><h4 style= 'color: white;'>Conclusion</h4></strong>
        The Car Price Prediction App is a powerful tool for individuals and businesses looking to evaluate 
        car prices efficiently. By leveraging machine learning, it offers a seamless experience in determining a 
        car's fair value, making the buying and selling process more transparent and informed.
    </div>
    """, unsafe_allow_html=True)


with tab2:
    st.write("<h4 style = 'color: black;'>Welcome to the Price Values tab!</h4>", unsafe_allow_html=True)
    a1, a2, a3 = st.columns(3)
    a4, a5, a6 = st.columns(3)
    a7, a8, a9 = st.columns(3)
    a10, a11, a12 = st.columns(3)
    a13, a14 = st.columns(2)
    
    with a1:
        city_select=st.selectbox("City",dropdown_options["city"])
        city=encoder_city.transform([[city_select]])[0][0]
    with a2:
        ft_select=st.selectbox("Fuel Type",dropdown_options["ft"])
        ft=encoder_ft.transform([[ft_select]])[0][0]
    with a3:
        bt_select=st.selectbox("Body Type",dropdown_options["bt"])
        bt=encoder_bt.transform([[bt_select]])[0][0]
    with a4:
        km=st.number_input("KM driven",min_value=10)
    with a5:
        transmission_select=st.selectbox("Transmission",dropdown_options["transmission"])
        transmission=encoder_transmission.transform([[transmission_select]])[0][0]
    with a6:
        ownerNo=st.number_input("No. of Owner's",min_value=1)
    with a7:
        oem_list=ml_df[ml_df["ft"]==ft_select]["oem"]
        oem_filtered=oem_list.unique().tolist()
        oem_select=st.selectbox("Manufacture Company",oem_filtered)
        oem=encoder_oem.transform([[oem_select]])[0][0]
    with a8:
        model_list=ml_df[ml_df["oem"]==oem_select]["model"]
        model_filtered=model_list.unique().tolist()
        model_select=st.selectbox("Car Model Name",model_filtered)
        model=encoder_model.transform([[model_select]])[0][0]
    with a9:
        modelYear=st.number_input("Car Manufacture Year",min_value=1900)
    with a10:
        variantName_list=ml_df[ml_df["model"]==model_select]["variantName"]
        variantName_filtered=variantName_list.unique().tolist()
        variantName_select=st.selectbox("Model Variant Name",variantName_filtered)
        variantName=encoder_variantName.transform([[variantName_select]])[0][0]
    with a11:
        Registration_Year=st.number_input("Car Registration Year",min_value=1900)
    with a12:
        InsuranceValidity_select=st.selectbox("Insurance Type",dropdown_options["Insurance Validity"])
        InsuranceValidity=encoder_Insurance_Validity.transform([[InsuranceValidity_select]])[0][0]
    with a13:
        Seats=st.number_input("Car seat capacity",min_value=4)
    with a14:
        EngineDisplacement=st.number_input("Engine (CC)",min_value=799)
        
    if st.button('Click Here!'):
        input_data = pd.DataFrame([city,ft,bt,km,transmission,ownerNo,oem,model,modelYear,variantName,Registration_Year,InsuranceValidity,Seats,EngineDisplacement])

        prediction = model_car.predict(input_data.values.reshape(1, -1))
                
        st.markdown("<h2 style='color: black;'>Car Price</h2>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='color: Green;'>₹ {prediction[0]:,.2f}</h3>", unsafe_allow_html=True)
