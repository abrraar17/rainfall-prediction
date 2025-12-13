import pandas as pd
import streamlit as st
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ------------------ Styling ------------------
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

# ------------------ Load Data ------------------
data = pd.read_csv("rainfall_data.csv")

X = data.drop("rainfall", axis=1)
y = data["rainfall"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------ Train Model ------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# ------------------ Header ------------------
st.title("☔ Rainfall Predictor")
st.markdown("### Machine Learning based Rain Prediction")
st.markdown("---")

# ------------------ Sidebar ------------------
with st.sidebar:
    st.markdown("### About")
    st.caption("Built by Muhammad Zia Uddin")
    st.write(
        "This model predicts the **chance of rainfall** using "
        "real-time weather parameters such as temperature, humidity, "
        "pressure, cloud cover, wind speed and visibility."
    )

# ------------------ Model Info ------------------
col1, col2 = st.columns(2)
col1.metric("Model Accuracy", f"{acc:.2%}")
col2.code("Random Forest Classifier")

# ------------------ PIN Code Input ------------------
st.markdown("### 📍 Enter PIN Code (India)")
pincode = st.text_input("Example: 500024")

# ------------------ PIN → Lat/Lon (FIXED) ------------------
def get_lat_lon_from_pincode(pincode):
    url = (
        "https://geocoding-api.open-meteo.com/v1/search"
        f"?postalcode={pincode}&country=IN"
    )
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    data = r.json()

    if "results" not in data or len(data["results"]) == 0:
        return None, None, None

    loc = data["results"][0]
    return loc["latitude"], loc["longitude"], loc["name"]

if not pincode:
    st.stop()

try:
    lat, lon, place = get_lat_lon_from_pincode(pincode)

    if lat is None:
        st.error("Invalid PIN code. Location not found.")
        st.stop()

    st.success(f"Location detected: {place}")

    # ------------------ Fetch Live Weather ------------------
    weather_api = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&current=temperature_2m,relative_humidity_2m,"
        "cloud_cover,windspeed_10m,pressure_msl,visibility"
    )

    resp = requests.get(weather_api, timeout=10)
    resp.raise_for_status()
    current = resp.json()["current"]

    pressure = current.get("pressure_msl", 1010)
    temperature = current.get("temperature_2m", 25)
    humidity = current.get("relative_humidity_2m", 60)
    windspeed = current.get("windspeed_10m", 10)
    cloudcover = current.get("cloud_cover", 50)
    visibility = round(current.get("visibility", 10000) / 1000)

    st.markdown("### 🌐 Live Weather Data")
    st.write(f"🌡️ Temperature: {temperature} °C")
    st.write(f"💧 Humidity: {humidity} %")
    st.write(f"🌬️ Wind Speed: {windspeed} km/h")
    st.write(f"🌥️ Cloud Cover: {cloudcover} %")
    st.write(f"📈 Pressure: {pressure} hPa")
    st.write(f"👁️ Visibility: {visibility} km")

    # ------------------ Prediction ------------------
    features = [[
        pressure,
        temperature,
        humidity,
        windspeed,
        cloudcover,
        visibility
    ]]

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][prediction]

    st.markdown("---")

    if prediction == 1:
        st.success(f"🌧️ Rain Expected with **{probability:.2%} confidence**")
    else:
        st.info(f"☀️ No Rain Expected with **{probability:.2%} confidence**")

    # ------------------ Explanation ------------------
    st.markdown("### 🤔 What does this confidence mean?")
    st.write(
        f"The model analyzed current weather conditions and found patterns "
        f"similar to **{probability:.0%}** of past situations where the outcome "
        f"was **{'rain' if prediction == 1 else 'no rain'}**.\n\n"
        "This is **not a guarantee**. Weather is chaotic — the model provides "
        "a **probability**, not a promise."
    )

except Exception as e:
    st.error(f"Error fetching weather data: {e}")



