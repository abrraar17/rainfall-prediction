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

FEATURE_COLUMNS = [
    "pressure",
    "temperature",
    "humidity",
    "windspeed",
    "cloudcover",
    "visibility"
]

X = data[FEATURE_COLUMNS]
y = data["rainfall"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))

# ------------------ UI ------------------
st.title("☔ Rainfall Predictor")
st.markdown("### Machine Learning based Rain Prediction")
st.markdown("---")

with st.sidebar:
    st.markdown("### About")
    st.caption("Built by Muhammad Zia Uddin")
    st.write(
        "This model predicts the **chance of rainfall** using real-time "
        "weather parameters such as temperature, humidity, pressure, "
        "cloud cover, wind speed and visibility."
    )

col1, col2 = st.columns(2)
col1.metric("Model Accuracy", f"{accuracy:.2%}")
col2.code("Random Forest Classifier")

# ------------------ PIN Code Input ------------------
st.markdown("### 📍 Enter PIN Code (India)")
pin_code = st.text_input("Example: 500024", max_chars=6)

if pin_code:
    try:
        # ---------- PIN → LAT/LON ----------
        geo_url = (
            "https://nominatim.openstreetmap.org/search"
            f"?postalcode={pin_code}&country=India&format=json"
        )
        geo_resp = requests.get(
            geo_url,
            headers={"User-Agent": "RainfallPredictorApp"}
        )
        geo_resp.raise_for_status()
        geo_data = geo_resp.json()

        if not geo_data:
            st.error("Invalid PIN code. Location not found.")
            st.stop()

        lat = geo_data[0]["lat"]
        lon = geo_data[0]["lon"]
        location_name = geo_data[0]["display_name"]

        # ---------- WEATHER ----------
        weather_url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            "&current=temperature_2m,relative_humidity_2m,"
            "cloud_cover,windspeed_10m,pressure_msl,visibility"
        )

        weather_resp = requests.get(weather_url)
        weather_resp.raise_for_status()
        current = weather_resp.json()["current"]

        input_data = pd.DataFrame([{
            "pressure": current["pressure_msl"],
            "temperature": current["temperature_2m"],
            "humidity": current["relative_humidity_2m"],
            "windspeed": current["windspeed_10m"],
            "cloudcover": current["cloud_cover"],
            "visibility": round(current["visibility"] / 1000)
        }])

        st.markdown("### 🌐 Live Weather Data")
        st.write(f"**Location**: {location_name}")
        for k, v in input_data.iloc[0].items():
            st.write(f"**{k.capitalize()}**: {v}")

        # ---------- Prediction ----------
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][prediction]

        st.markdown("---")

        if prediction == 1:
            st.success(
                f"🌧️ **Rain Expected** with {probability:.2%} confidence"
            )
        else:
            st.info(
                f"☀️ **No Rain Expected** with {probability:.2%} confidence"
            )

        # ---------- Explanation (MOVED BELOW) ----------
        st.markdown(
            "**What does this mean?**\n\n"
            "The confidence value represents how often rainfall occurred "
            "under similar weather conditions in historical data. "
            "It does **not** guarantee rain."
        )

    except Exception as e:
        st.error(f"Error fetching weather data: {e}")
