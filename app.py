import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import plotly.express as px
import zipfile

st.set_page_config(page_title="Nigeria Rainfall Dashboard", layout="wide")

# ---------------------------------------------
# Load dataset from ZIP
# ---------------------------------------------
@st.cache_data
def load_data():
    with zipfile.ZipFile("Nigeria_Weather_Synthetic_2000_2024.zip") as z:
        with z.open("Nigeria_Weather_Synthetic_2000_2024.csv") as f:
            df = pd.read_csv(f, parse_dates=["date"])
    return df

df = load_data()

# ---------------------------------------------
# Train model inside Streamlit Cloud
# ---------------------------------------------
@st.cache_resource
def train_model():
    features = [
        "rainfall_mm", "temp_min", "temp_max", "temp_avg",
        "humidity", "wind_speed", "pressure", "sunshine_hours",
        "month", "dayofyear"
    ]

    df["month"] = df["date"].dt.month
    df["dayofyear"] = df["date"].dt.dayofyear

    X = df[features]
    y = df["RainTomorrow"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.25, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        random_state=42
    )
    model.fit(X_train, y_train)

    return model, scaler

model, scaler = train_model()

# ---------------------------------------------
# Sidebar status
# ---------------------------------------------
with st.sidebar:
    st.success("Dataset loaded — 37 states")
    st.success("Model trained in cloud (OK)")

# ---------------------------------------------
# UI Header
# ---------------------------------------------
st.title("Nigeria Weather Dashboard")
st.write("Choose a state, inspect climate trends, and see rainfall predictions.")

# ---------------------------------------------
# State & date selection
# ---------------------------------------------
states = sorted(df["state"].unique())
state_sel = st.selectbox("Select state", states)

date_sel = st.date_input("Select date", df["date"].min())

df_state = df[df["state"] == state_sel]

# ---------------------------------------------
# Today's metrics
# ---------------------------------------------
today_row = df_state[df_state["date"] == pd.to_datetime(date_sel)]

if today_row.empty:
    st.warning("No data available for this date.")
else:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Wind Speed", f"{today_row['wind_speed'].values[0]:.2f} m/s")
    col2.metric("Rain Today", f"{today_row['rainfall_mm'].values[0]:.1f} mm")
    col3.metric("Pressure", f"{today_row['pressure'].values[0]:.1f} hPa")
    col4.metric("Humidity", f"{today_row['humidity'].values[0]:.1f} %")

# ---------------------------------------------
# Prediction input section
# ---------------------------------------------
st.subheader("Prediction inputs")

colA, colB, colC = st.columns(3)

rain = colA.number_input("Rain Today (mm)", 0.0, 500.0, 0.0)
tmin = colA.number_input("Temp Min (°C)", -5.0, 50.0, 22.0)
tmax = colB.number_input("Temp Max (°C)", -5.0, 60.0, 30.0)
wind = colB.number_input("Wind Speed (m/s)", 0.0, 50.0, 3.0)
humidity = colC.number_input("Humidity (%)", 0.0, 100.0, 70.0)
pressure = colC.number_input("Pressure (hPa)", 900.0, 1050.0, 1010.0)

sunshine = 6.0
month_val = pd.to_datetime(date_sel).month
dayofyear_val = pd.to_datetime(date_sel).dayofyear

# ---------------------------------------------
# Run prediction
# ---------------------------------------------
if st.button("Predict"):
    input_data = np.array([[
        rain, tmin, tmax, (tmin+tmax)/2,
        humidity, wind, pressure, sunshine,
        month_val, dayofyear_val
    ]])

    input_scaled = scaler.transform(input_data)
    prob = model.predict_proba(input_scaled)[0][1]

    st.success(f"Probability of rain tomorrow: **{prob*100:.2f}%**")

# ---------------------------------------------
# Weekly overview chart
# ---------------------------------------------
st.subheader("Weekly Overview")

df_week = df_state.tail(30)
fig = px.line(df_week, x="date", y="temp_avg", title="Temperature Trend (Recent Days)")
st.plotly_chart(fig, use_container_width=True)
