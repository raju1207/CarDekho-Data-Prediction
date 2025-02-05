import streamlit as st
import pickle
from PIL import Image
import numpy as np

# Load the model
path = r"C:\Users\Raju\OneDrive\Desktop\PROJECTS\Car-Analysis\Car_Sales.pkl"
with open(path, "rb") as file:
    model = pickle.load(file)

# Add color to the title
st.markdown("<h1 style='color: black;'>Car Price Prediction</h1>", unsafe_allow_html=True)


input_features = ["Max Power", "Year of Manufacture", "Width", "Kerb Weight", "km", "Acceleration", "Mileage", "Wheel Base", "city"]

# Define label encoding for cities
city_mapping = {
    "Bangalore": 0,
    "Chennai": 1,
    "Delhi": 2,
    "Hyderabad": 3,
    "Jaipur": 4,
    "Kolkata": 5
}

# Mapping the values
year_mapping = { str(i): i - 1996 for i in range(1996, 2026)}
max_power = { str(i): i - 40 for i in range(40, 210)}
width = { str(i): i - 1000 for i in range(1000, 2600)}
wheel_base = { str(i): i - 2000 for i in range(2000, 5000)}
kerb_weight = { str(i): i - 500 for i in range(500, 4000)}
km_driven = { str(i): i - 1000 for i in range(1000, 101000)}
acceleration = { str(i): i - 0 for i in range(0, 21)}
mileage = {str(i): i - 5 for i in range(5, 151)}

st.image(r"C:\Users\Raju\OneDrive\Desktop\PROJECTS\Car-Analysis\Data\Scripts\car.image.jpg")
st.sidebar.header("Car Price Prediction")

params = {}
params["city"] = st.sidebar.selectbox("City", list(city_mapping.keys()))
params["Year of Manufacture"] = st.sidebar.selectbox("Year of Manufacture", list(year_mapping.keys()))
params["Max Power"] = st.sidebar.selectbox("Max Power (in HP)", list(max_power.keys()))
params["Width"] = st.sidebar.selectbox("Width (in mm)", list(width.keys()))
params["Wheel Base"] = st.sidebar.selectbox("Wheel Base (in mm)", list(wheel_base.keys()))
params["Kerb Weight"] = st.sidebar.selectbox("Kerb Weight (in kg)", list(kerb_weight.keys()))
params["km"] = st.sidebar.selectbox("km driven (in km)", list(km_driven.keys()))
params["Acceleration"] = st.sidebar.selectbox("Acceleration (0-100 km/h in sec)", list(acceleration.keys()))
params["Mileage"] = st.sidebar.selectbox("Mileage (KM/L)", list(mileage.keys()))

# Convert city to numeric
params["city"] = city_mapping[params["city"]]

# Prepare input data
input_data = np.array([params[feature] for feature in input_features]).reshape(1, -1)

# Predict the price
predicted_price = model.predict(input_data)[0]


st.write(f"The Predicted Price is: ₹{predicted_price:.2f} lakhs")
st.button("Result")
