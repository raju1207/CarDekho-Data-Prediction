import streamlit as st
import pickle
import numpy as np
import pandas as pd
import base64
import re
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
from datetime import datetime
import streamlit.components.v1 as components
from nlp import chatbot_response


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

st.markdown("""
    <style>
    /* Make all tab labels bold */
    div[data-testid="stTabs"] button {
        font-weight: bold !important;
        color: black;
    }
    </style>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Home", "Price Values", "Chatbot"])

with tab1:
    st.write("<h4 style = 'color: black;'><strong>Welcome to the Home tab!</strong></h4>", unsafe_allow_html=True)
    st.markdown("""
    <div style='color: black; font-size: 20px;'>
        <strong><h4 style= 'color: white; font-size: 22px'><strong>Introduction</strong></h4></strong>
        In the rapidly evolving automotive market, determining the right price for a vehicle is crucial 
        for both buyers and sellers. The Car Price Prediction App provides an intelligent solution to 
        estimate car prices based on key parameters using machine learning models. This tool helps users 
        make data-driven decisions by leveraging historical data and predictive analytics.<br>
        
    <strong><h4 style= 'color: white; font-size: 20px'><strong>Key Features</strong></h4></strong>
        <ul>
            <li>User-Friendly Interface: Simple and interactive Streamlit-based UI.</li>
            <li>Machine Learning Model: Utilizes an advanced regression model (XGBRegressor) trained 
            on a vast dataset of car prices.</li>
            <li>Feature Inputs: Users can enter details like car brand, model, manufacturing year, fuel type, 
            transmission, and other relevant attributes.</li>
            <li>Real-Time Predictions: Provides instant car price estimates based on input parameters.</li>
            <li>Comparison Tool: Allows users to compare multiple cars for better decision-making.</li>
        </ul>
        
    <strong><h4 style= 'color: white; font-size: 20px'><strong>Conclusion</strong></h4></strong>
        The Car Price Prediction App is a powerful tool for individuals and businesses looking to evaluate 
        car prices efficiently. By leveraging machine learning, it offers a seamless experience in determining a 
        car's fair value, making the buying and selling process more transparent and informed.
    </div>
    """, unsafe_allow_html=True)

with tab2:
    st.write("<h4 style = 'color: black;'>Welcome to the Price Values tab!</h4>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    col4, col5, col6 = st.columns(3)
    col7, col8, col9 = st.columns(3)
    col10, col11, col12 = st.columns(3)
    col13, col14 = st.columns(2)

    with col1:
        city_input = st.selectbox("**City**", dropdown_options["city"])
        city = encoder_city.transform([[city_input]])[0][0]

    with col2:
        ft_input = st.selectbox("**Fuel Type**", dropdown_options["ft"])
        ft = encoder_ft.transform([[ft_input]])[0][0]

    with col3:
        bt_input = st.selectbox("**Body Type**", dropdown_options["bt"])
        bt = encoder_bt.transform([[bt_input]])[0][0]

    with col4:
        km = st.number_input("**KM driven**", min_value=10)

    with col5:
        trans_input = st.selectbox("**Transmission**", dropdown_options["transmission"])
        transmission = encoder_transmission.transform([[trans_input]])[0][0]

    with col6:
        ownerNo = st.number_input("**No. of Owners**", min_value=1)

    with col7:
        oem_options = ml_df[ml_df["ft"] == ft_input]["oem"].dropna().unique().tolist()
        oem_input = st.selectbox("**Manufacture Company**", oem_options)
        oem = encoder_oem.transform([[oem_input]])[0][0]

    with col8:
        model_options = ml_df[ml_df["oem"] == oem_input]["model"].dropna().unique().tolist()
        model_input = st.selectbox("**Car Model Name**", model_options)
        model = encoder_model.transform([[model_input]])[0][0]

    with col9:
        current_year = datetime.now().year
        modelYear = st.number_input("**Car Manufacture Year**", min_value=1900, max_value=current_year)

    with col10:
        variant_options = ml_df[ml_df["model"] == model_input]["variantName"].dropna().unique().tolist()
        variant_input = st.selectbox("**Model Variant Name**", variant_options)
        variantName = encoder_variantName.transform([[variant_input]])[0][0]

    with col11:
        Registration_Year = st.number_input("**Registration Year**", min_value=1900, max_value=current_year)

    with col12:
        ins_input = st.selectbox("**Insurance Type**", dropdown_options["Insurance Validity"])
        InsuranceValidity = encoder_Insurance_Validity.transform([[ins_input]])[0][0]

    with col13:
        Seats = st.number_input("**Seating Capacity**", min_value=2)

    with col14:
        EngineDisplacement = st.number_input("**Engine CC**", min_value=600)


    if st.button('**Click Here!**'):
        
        input_data = pd.DataFrame([[city, ft, bt, km, transmission, ownerNo, oem, model, modelYear,
                            variantName, Registration_Year, InsuranceValidity, Seats, EngineDisplacement]],
                          columns=['city', 'ft', 'bt', 'km', 'transmission', 'ownerNo', 'oem',
                                   'model', 'modelYear', 'variantName', 'Registration_Year',
                                   'Insurance Validity', 'Seats', 'EngineDisplacement'])
        

        prediction = model_car.predict(input_data.values.reshape(1, -1))
                
        st.markdown("<h2 style='color: black;'>Car Price</h2>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='color: Green;'>₹ {prediction[0]:,.2f}</h3>", unsafe_allow_html=True)


with tab3:

    # # Load car data
    # df = pd.read_excel("All_Cities_ML_Data.xlsx")

    # st.write("<h4 style = 'color: black;'>🚗 CarBot-Car Finder by Brand</h4>", unsafe_allow_html=True)

    # # Input
    # user_input = st.text_input("🔍 Enter a car details", "")

    # # Clear button
    # if st.button("🧹 Clear"):
    #     st.rerun()

    # # On input
    # if user_input:
    #     query = user_input.lower()
    #     matched_brand = None

    #     for brand in df["oem"].dropna().unique():
    #         if isinstance(brand, str) and brand.lower() in query:
    #             matched_brand = brand
    #             break

    #     if matched_brand:
    #         st.success(f"Showing **{matched_brand}** Car Details:")
    #         brand_df = df[df["oem"] == matched_brand]
    #         cities = brand_df["city"].dropna().unique()


    #         for city in cities:
    #             car = brand_df[brand_df["city"] == city].head(1)
    #             if not car.empty:
    #                 car_info = car.iloc[0]

    #             # Extract and clean fields
    #             model = str(car_info.get("model", "N/A"))
    #             variant = str(car_info.get("variantName", ""))
    #             location = str(car_info.get("city", "N/A"))
    #             fuel = str(car_info.get("ft", "N/A"))
    #             transmission = str(car_info.get("transmission", "N/A"))
    #             engine = str(car_info.get("Engine Displacement", "N/A"))
    #             reg_year = str(car_info.get("Registration Year", "N/A"))
    #             owner = str(car_info.get("ownerNo", "N/A"))
    #             try:
    #                 price = f"₹{int(str(car_info.get('price', '0')).replace(',', '').replace('₹', '')):,}"
    #             except:
    #                 price = str(car_info.get("price", "N/A"))

    #             try:
    #                 km = f"{int(str(car_info.get('km', '0')).replace(',', '')):,} km"
    #             except:
    #                 km = "N/A"

    #             # try:
    #             #     seats = int(car_info.get("Seats", 0))
    #             # except:
    #             #     seats = "N/A"
                
    #             raw_seats = car_info.get("Seats", "")
    #             if pd.notna(raw_seats):
    #                 import re
    #                 match = re.search(r"\d+", str(raw_seats))
    #                 seats = int(match.group()) if match else "Not Specified"
    #             else:
    #                 seats = "Not Specified"


    #             st.markdown(f"""
    #             <div style="border:1px solid #ccc; border-radius:5px; padding:5px; margin-bottom:px; background-color:#f9f9f9;">
    #                 <h4 style="color:#007BFF;">🚘 {model} {variant}</h4>
    #                 <p>📍 <strong>Location:</strong> {location}</p>
    #                 <p>💰 <strong>Price:</strong> {price}</p>
    #                 <p>⛽ <strong>Fuel Type:</strong> {fuel} &nbsp;&nbsp; ⚙️ <strong>Transmission:</strong> {transmission}</p>
    #                 <p>🛣️ <strong>Kilometers Driven:</strong> {km} &nbsp;&nbsp; 🛋️ <strong>Seats:</strong> {seats}</p>
    #                 <p>📅 <strong>Registration Year:</strong> {reg_year} &nbsp;&nbsp; 🪪 <strong>Owner Number:</strong> {owner}</p>
    #                 <p>🔧 <strong>Engine Displacement:</strong> {engine}</p>
    #             </div>
    #             """, unsafe_allow_html=True)

                    
    df = pd.read_excel("All_Cities_ML_Data.xlsx")
    st.markdown("<h4 style='color:black;'>🚗 CarBot - Smart Car Finder</h4>", unsafe_allow_html=True)
    
    user_input = st.text_input("🔍 Enter a query like 'Honda cars in Chennai' or 'Tell me about Hyundai'", "")

    if st.button("🧹 Clear"):
        st.rerun()

    if user_input:
        query = user_input.lower()

        # Extract brand and city from the query using simple matching
        matched_brand = None
        matched_city = None

        # Match brand
        for brand in df["oem"].dropna().unique():
            if isinstance(brand, str) and brand.lower() in query:
                matched_brand = brand
                break

        # Match city
        for city in df["city"].dropna().unique():
            if isinstance(city, str) and city.lower() in query:
                matched_city = city
                break

        if matched_brand:
            if matched_city:
                filtered_df = df[(df["oem"] == matched_brand) & (df["city"] == matched_city)]
                st.success(f"Showing top {min(10, len(filtered_df))} **{matched_brand}** cars in **{matched_city}**")
            else:
                filtered_df = df[df["oem"] == matched_brand]
                st.success(f"Showing top {min(10, len(filtered_df))} **{matched_brand}** cars across all cities")

            # Limit to top 10 results
            top_cars = filtered_df.head(10)

            if top_cars.empty:
                st.warning("No matching cars found. Try a different query.")
            else:
                for idx, car_info in top_cars.iterrows():
                    model = str(car_info.get("model", "N/A"))
                    variant = str(car_info.get("variantName", ""))
                    location = str(car_info.get("city", "N/A"))
                    fuel = str(car_info.get("ft", "N/A"))
                    transmission = str(car_info.get("transmission", "N/A"))
                    engine = str(car_info.get("Engine Displacement", "N/A"))
                    reg_year = str(car_info.get("Registration Year", "N/A"))
                    owner = str(car_info.get("ownerNo", "N/A"))

                    try:
                        price = f"₹{int(str(car_info.get('price', '0')).replace(',', '').replace('₹', '')):,}"
                    except:
                        price = str(car_info.get("price", "N/A"))

                    try:
                        km = f"{int(str(car_info.get('km', '0')).replace(',', '')):,} km"
                    except:
                        km = "N/A"

                    raw_seats = car_info.get("Seats", "")
                    seats = int(re.search(r"\d+", str(raw_seats)).group()) if pd.notna(raw_seats) and re.search(r"\d+", str(raw_seats)) else "Not Specified"

                    st.markdown(f"""
                    <div style="border:1px solid #ccc; border-radius:5px; padding:10px; margin-bottom:10px; background-color:#f9f9f9;">
                        <h4 style="color:#007BFF;">🚘 {model} {variant}</h4>
                        <p>📍 <strong>Location:</strong> {location}</p>
                        <p>💰 <strong>Price:</strong> {price}</p>
                        <p>⛽ <strong>Fuel Type:</strong> {fuel} &nbsp;&nbsp; ⚙️ <strong>Transmission:</strong> {transmission}</p>
                        <p>🛣️ <strong>Kilometers Driven:</strong> {km} &nbsp;&nbsp; 🛋️ <strong>Seats:</strong> {seats}</p>
                        <p>📅 <strong>Registration Year:</strong> {reg_year} &nbsp;&nbsp; 🪪 <strong>Owner Number:</strong> {owner}</p>
                        <p>🔧 <strong>Engine Displacement:</strong> {engine}</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning("❌ No matching brand found. Try again like: 'Tell me about Hyundai cars in Bangalore'.")


# with tab4:
#     st.title("Car Resale Price EDA")

#     # Load your dataframe
#     ml_df = pd.read_excel("All_Cities_ML_Data.xlsx")

#     # Show basic info
#     st.subheader("🚗 **Dataset Overview**")

#     # Dropdown to select city
#     city_list = ["All"] + sorted(ml_df["city"].unique().tolist())
#     selected_city = st.selectbox("**Select City to View Data**", city_list)

#     # Filter dataframe based on selection
#     if selected_city == "All":
#         filtered_df = ml_df
#     else:
#         filtered_df = ml_df[ml_df["city"] == selected_city]

#     # Show the filtered dataframe
#     st.dataframe(filtered_df.head(50))

#     st.write(f"**Showing** {filtered_df.shape[0]} **rows for city**: **{selected_city}**")

#     # Data overview: missing values and histogram
#     st.subheader("📊 Data Distribution & Missing Value Analysis")

#     # Show missing values per column
#     st.write(ml_df.isnull().sum())

#     # Column selection for histogram
#     column = st.selectbox("**Choose a numeric column for histogram**", ml_df.select_dtypes(include='number').columns)
#     fig, ax = plt.subplots()
#     sns.histplot(ml_df[column], kde=True, ax=ax)
#     st.pyplot(fig)

#     # Correlation heatmap
#     st.subheader("**Correlation Heatmap**")
#     fig2, ax2 = plt.subplots(figsize=(10, 6))
#     sns.heatmap(ml_df.select_dtypes(include='number').corr(), annot=True, cmap="coolwarm", ax=ax2)
#     st.pyplot(fig2)

#     # Boxplot for outliers
#     st.subheader("**Boxplot for Outliers**")
#     box_col = st.selectbox("**Select a numeric column**", ml_df.select_dtypes(include='number').columns)
#     fig3, ax3 = plt.subplots()
#     sns.boxplot(x=ml_df[box_col], ax=ax3)
#     st.pyplot(fig3)

#     # Scatter vs price
#     st.subheader("**Price vs Feature Scatterplot**")
#     scatter_col = st.selectbox("**Select a feature**", ml_df.columns.drop("price"))
#     fig4, ax4 = plt.subplots()
#     sns.scatterplot(x=ml_df[scatter_col], y=ml_df["price"], ax=ax4)
#     st.pyplot(fig4)
