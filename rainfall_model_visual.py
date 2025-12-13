import pandas as pd
import numpy as np
import streamlit as st
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# --- Custom Styling for Background ---
def local_css():
    st.markdown("""
        <style>
        .stApp {
            background-color: #121212;
            color: white;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #00B4D8;
        }
        </style>
        """, unsafe_allow_html=True)

local_css()

# --- Load and Prepare Data ---
data = pd.read_csv("rainfall_data.csv")  # Ensure your CSV is in same folder
X = data.drop("rainfall", axis=1)
y = data["rainfall"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Train Model ---
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# --- UI Title & Header ---
st.title("☔ Rainfall Predictor")
st.markdown("### Powered by Real-Time Weather Data + Machine Learning")
st.markdown("---")

# --- Sidebar Info ---
with st.sidebar:
    st.image("https://i.imgur.com/9A8aXyQ.png", width=150)
    st.markdown("### 🌦️ Smart Rain Prediction")
    st.caption("built by:Muhammad zia uddin")


# --- Model Accuracy ---
col1, col2 = st.columns(2)
col1.metric("Model Accuracy", f"{acc:.2%}")
col2.code("Random Forest Classifier")

# --- City Selection Dropdown ---
st.markdown("### 📍 Choose a City")
cities = {
    "Hyderabad": "17.3850,78.4867",
    "Delhi": "28.6139,77.2090",
    "Mumbai": "19.0760,72.8777",
    "Bangalore": "12.9716,77.5946",
    "Chennai": "13.0827,80.2707"
}
selected_city = st.selectbox("Select a city", list(cities.keys()))

# --- Fetch Live Weather ---
latlon = cities[selected_city]
weather_api = f"https://api.open-meteo.com/v1/forecast?latitude={latlon.split(',')[0]}&longitude={latlon.split(',')[1]}&current=temperature_2m,relative_humidity_2m,cloud_cover,windspeed_10m,pressure_msl,visibility"
try:
    resp = requests.get(weather_api)
    resp.raise_for_status()
    current = resp.json()["current"]

    # Extract features
    features = [
        current.get("pressure_msl", 1010),
        current.get("temperature_2m", 25),
        current.get("relative_humidity_2m", 60),
        current.get("windspeed_10m", 10),
        current.get("cloud_cover", 50),
        round(current.get("visibility", 10000)/1000)  # km approx
    ]

    st.markdown("### 🌐 Live Weather Data")
    st.write(f"**City**: {selected_city}")
    st.write(f"🌡️ Temperature: {features[1]} °C")
    st.write(f"💧 Humidity: {features[2]} %")
    st.write(f"🌬️ Wind Speed: {features[3]} km/h")
    st.write(f"🌥️ Cloud Cover: {features[4]} %")
    st.write(f"📈 Pressure: {features[0]} hPa")
    st.write(f"👁️ Visibility: {features[5]} km")

    # Prediction
    prediction = model.predict([features])[0]
    prob = model.predict_proba([features])[0][prediction]

    st.markdown("---")
    if prediction == 1:
        st.success(f"🌧️ **Rain Expected** with {prob:.2%} confidence")
    else:
        st.info(f"☀️ **No Rain** with {prob:.2%} confidence")

except Exception as e:
    st.error(f"Error fetching weather data: {e}")


