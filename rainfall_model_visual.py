import pandas as pd
import numpy as np
import streamlit as st
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# --- Custom Styling ---
def local_css():
    st.markdown("""
        <style>
        .stApp {
            background-color: #121212;
            color: white;
        }
        h1, h2, h3 {
            color: #00B4D8;
        }
        </style>
    """, unsafe_allow_html=True)

local_css()

# --- Load and Prepare Data ---
data = pd.read_csv("rainfall_data.csv")
X = data.drop("rainfall", axis=1)
y = data["rainfall"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- Train Model ---
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
acc = accuracy_score(y_test, model.predict(X_test))

# --- UI ---
st.title("☔ Rainfall Predictor")
st.markdown("### Machine Learning based Rain Prediction")
st.markdown("---")

# --- Sidebar ---
with st.sidebar:
    st.markdown("### 🌧️ About")
    st.caption("Built by Muhammad Zia Uddin")
    st.markdown("""
    This model predicts **chance of rainfall**
    using real-time weather parameters.
    """)

# --- Model Accuracy ---
col1, col2 = st.columns(2)
col1.metric("Model Accuracy", f"{acc:.2%}")
col2.code("Random Forest Classifier")

# --- PIN CODE INPUT ---
st.markdown("### 📍 Enter PIN Code (India)")
pincode = st.text_input("Example: 500024")

if pincode:
    try:
        # --- Geocoding using Open-Meteo (NO API KEY) ---
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={pincode}&count=1&language=en&format=json"
        geo_resp = requests.get(geo_url).json()

        if "results" not in geo_resp:
            st.error("Invalid PIN code. Location not found.")
            st.stop()

        lat = geo_resp["results"][0]["latitude"]
        lon = geo_resp["results"][0]["longitude"]
        location_name = geo_resp["results"][0]["name"]

        # --- Weather API ---
        weather_api = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            f"&current=temperature_2m,relative_humidity_2m,"
            f"cloud_cover,windspeed_10m,pressure_msl,visibility"
        )

        resp = requests.get(weather_api)
        current = resp.json()["current"]

        features = [
            current.get("pressure_msl", 1010),
            current.get("temperature_2m", 25),
            current.get("relative_humidity_2m", 60),
            current.get("windspeed_10m", 10),
            current.get("cloud_cover", 50),
            round(current.get("visibility", 10000) / 1000)
        ]

        # --- Display Weather ---
        st.markdown("### 🌐 Live Weather Data")
        st.write(f"**Location**: {location_name} ({pincode})")
        st.write(f"🌡️ Temperature: {features[1]} °C")
        st.write(f"💧 Humidity: {features[2]} %")
        st.write(f"🌬️ Wind Speed: {features[3]} km/h")
        st.write(f"🌥️ Cloud Cover: {features[4]} %")
        st.write(f"📈 Pressure: {features[0]} hPa")
        st.write(f"👁️ Visibility: {features[5]} km")

        # --- Prediction ---
        prediction = model.predict([features])[0]
        probability = model.predict_proba([features])[0][1]

        st.markdown("---")

        if prediction == 1:
            st.success(f"🌧️ Rain Possible — **{probability*100:.1f}% likelihood**")
        else:
            st.info(f"☀️ Rain Unlikely — **{(1-probability)*100:.1f}% confidence**")

        # --- Explanation ---
        st.markdown("### ℹ️ What does this mean?")
        st.markdown(f"""
- The model estimates the **probability of rain**, not a guarantee.
- **{probability*100:.1f}%** means:  
  _Under similar weather conditions in the past, rain occurred this often._
- Weather is chaotic — real-world rain **can still differ**.
        """)

    except Exception as e:
        st.error(f"Error: {e}")



