import pandas as pd
import streamlit as st
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ------------------ UI STYLE ------------------
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

# ------------------ LOAD DATA ------------------
data = pd.read_csv("rainfall_data.csv")

X = data.drop("rainfall", axis=1)
y = data["rainfall"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------ TRAIN MODEL ------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

acc = accuracy_score(y_test, model.predict(X_test))

# ------------------ HEADER ------------------
st.title("☔ Rainfall Predictor")
st.markdown("### Machine Learning based Rain Prediction")
st.markdown("---")

# ------------------ SIDEBAR ------------------
with st.sidebar:
    st.markdown("### About")
    st.caption("Built by Muhammad Zia Uddin")
    st.write(
        "This model predicts the **chance of rainfall** using real-time "
        "weather parameters such as temperature, humidity, pressure, "
        "cloud cover, wind speed and visibility."
    )

# ------------------ MODEL INFO ------------------
col1, col2 = st.columns(2)
col1.metric("Model Accuracy", f"{acc:.2%}")
col2.code("Random Forest Classifier")

# ------------------ PIN INPUT ------------------
st.markdown("### 📍 Enter PIN Code (India)")
pin_code = st.text_input("Example: 500024", max_chars=6)

if pin_code:
    try:
        # ---------- STEP 1: PIN → LAT/LON ----------
        geo_url = (
            "https://geocoding-api.open-meteo.com/v1/search"
            f"?postal_code={pin_code}&country=IN&count=1"
        )

        geo_resp = requests.get(geo_url, timeout=10)
        geo_resp.raise_for_status()
        geo_data = geo_resp.json()

        if "results" not in geo_data:
            st.error("Invalid PIN code. Location not found.")
            st.stop()

        location = geo_data["results"][0]
        lat = location["latitude"]
        lon = location["longitude"]
        place = location.get("name", "Unknown location")

        # ---------- STEP 2: WEATHER ----------
        weather_url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            "&current=temperature_2m,relative_humidity_2m,"
            "cloud_cover,windspeed_10m,pressure_msl,visibility"
        )

        weather_resp = requests.get(weather_url, timeout=10)
        weather_resp.raise_for_status()
        current = weather_resp.json()["current"]

        features = [
            current["pressure_msl"],
            current["temperature_2m"],
            current["relative_humidity_2m"],
            current["windspeed_10m"],
            current["cloud_cover"],
            round(current["visibility"] / 1000)
        ]

        # ---------- DISPLAY WEATHER ----------
        st.markdown("### 🌐 Live Weather Data")
        st.write(f"**Location:** {place} ({pin_code})")
        st.write(f"🌡️ Temperature: {features[1]} °C")
        st.write(f"💧 Humidity: {features[2]} %")
        st.write(f"🌬️ Wind Speed: {features[3]} km/h")
        st.write(f"🌥️ Cloud Cover: {features[4]} %")
        st.write(f"📈 Pressure: {features[0]} hPa")
        st.write(f"👁️ Visibility: {features[5]} km")

        # ---------- PREDICTION ----------
        prediction = model.predict([features])[0]
        probability = model.predict_proba([features])[0][prediction]

        st.markdown("---")

        if prediction == 1:
            st.success(
                f"🌧️ **Rain Expected**\n\n"
                f"Confidence: **{probability:.2%}**\n\n"
                "_This means: based on current weather patterns, "
                "the model leans toward rainfall — not a guarantee._"
            )
        else:
            st.info(
                f"☀️ **No Rain Expected**\n\n"
                f"Confidence: **{probability:.2%}**"
            )

    except requests.exceptions.RequestException:
        st.error("Weather service temporarily unavailable. Try again later.")



