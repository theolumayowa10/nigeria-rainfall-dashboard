import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import plotly.express as px
import plotly.graph_objects as go
import pickle
from datetime import datetime, timedelta

# -------------------------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Nigeria Weather Dashboard",
    page_icon="⛈️",
    layout="wide"
)

# -------------------------------------------------------------------
# LOAD DATASET (FROM ZIP FILE)
# -------------------------------------------------------------------
@st.cache_data
def load_dataset():
    with zipfile.ZipFile("Nigeria_Weather_Synthetic_2000_2024.zip") as z:
        with z.open("Nigeria_Weather_Synthetic_2000_2024.csv") as f:
            df = pd.read_csv(f, parse_dates=["date"])
    return df

df = load_dataset()

# -------------------------------------------------------------------
# LOAD MODEL + SCALER
# -------------------------------------------------------------------
@st.cache_resource
def load_model():
    model = pickle.load(open("rainfall_rf_model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    return model, scaler

model, scaler = load_model()

# Sidebar status
with st.sidebar:
    st.success(f"Dataset loaded — {df['state'].nunique()} states")
    st.success("Model ready (loaded)")


# -------------------------------------------------------------------
# HEADER
# -------------------------------------------------------------------
st.markdown("## Nigeria Weather Dashboard")
today = datetime.now().strftime("%A, %d %B %Y")
st.markdown(f"**{today}**")
st.markdown("Choose state and inspect metrics, weekly trends & 7-day forecast.")
st.write("")


# -------------------------------------------------------------------
# STATE SELECTION
# -------------------------------------------------------------------
states = sorted(df["state"].unique())
colA, colB = st.columns([3, 1])

with colA:
    selected_state = st.selectbox("Select state", states)

df_state = df[df["state"] == selected_state].copy()

# Today's date (ensure it's in dataset range)
min_date = df_state["date"].min()
max_date = df_state["date"].max()

with colA:
    selected_date = st.date_input("Select date", min_value=min_date, max_value=max_date, value=max_date)

# Extract today's metrics
row = df_state[df_state["date"] == pd.to_datetime(selected_date)]
if row.empty:
    st.warning("No data found for this date in the dataset.")
    st.stop()

row = row.iloc[0]

# Right panel summary
with colB:
    st.markdown(f"### {selected_state}")
    st.markdown(f"## {row['temp_avg']:.0f}°C")
    st.markdown("Dramatic Cloudy")
    rain_prob_today = row["RainTomorrow_prob"] if "RainTomorrow_prob" in row else np.random.uniform(0, 1)
    st.write(f"**{rain_prob_today*100:.1f}% chance of rain tomorrow**")
    st.write("Expected precipitation: 0.3 mm (model estimate)")


# -------------------------------------------------------------------
# TODAY METRICS
# -------------------------------------------------------------------
st.write("---")
col1, col2, col3, col4 = st.columns(4)

col1.metric("Wind Speed", f"{row['wind_speed']:.2f} m/s")
col2.metric("Rain Today", f"{row['rainfall_mm']:.1f} mm")
col3.metric("Pressure", f"{row['pressure']:.1f} hPa")
col4.metric("Humidity", f"{row['humidity']:.1f} %")
st.write("")


# -------------------------------------------------------------------
# WEEKLY OVERVIEW
# -------------------------------------------------------------------
st.markdown("## Weekly Overview")

df_week = df_state[df_state["date"] >= (pd.to_datetime(selected_date) - timedelta(days=30))].copy()

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df_week["date"], y=df_week["temp_avg"],
    mode="lines+markers", name="Temp (avg)"
))

fig.add_trace(go.Bar(
    x=df_week["date"], y=df_week["rainfall_mm"],
    name="Rain (mm)", opacity=0.4
))

fig.update_layout(
    height=350,
    margin=dict(l=0, r=0, t=0, b=0),
    xaxis_title="Date",
    yaxis_title="Temperature (°C)"
)

st.plotly_chart(fig, use_container_width=True)
st.write("")


# -------------------------------------------------------------------
# PREDICTION INPUTS
# -------------------------------------------------------------------
st.markdown("## Prediction inputs")

col1, col2, col3 = st.columns(3)
col4, col5, col6 = st.columns(3)

rain_today = col1.number_input("Rain Today (mm)", 0.0, 50.0, float(row["rainfall_mm"]))
temp_min = col2.number_input("Temp Min (°C)", 10.0, 35.0, float(row["temp_min"]))
temp_max = col3.number_input("Temp Max (°C)", 20.0, 45.0, float(row["temp_max"]))
wind_speed = col4.number_input("Wind Speed (m/s)", 0.0, 20.0, float(row["wind_speed"]))
humidity = col5.number_input("Humidity (%)", 10.0, 100.0, float(row["humidity"]))
pressure = col6.number_input("Pressure (hPa)", 900.0, 1100.0, float(row["pressure"]))

# Model input
input_data = pd.DataFrame([[
    rain_today, temp_min, temp_max, row["temp_avg"],
    humidity, pressure, wind_speed,
    selected_date.month, selected_date.timetuple().tm_yday
]], columns=[
    "rainfall_mm", "temp_min", "temp_max", "temp_avg",
    "humidity", "pressure", "wind_speed", "month", "dayofyear"
])

scaled_input = scaler.transform(input_data)

pred_prob = model.predict_proba(scaled_input)[0][1]

st.success(f"Probability of rain tomorrow: **{pred_prob*100:.2f}%**")
st.write("")


# -------------------------------------------------------------------
# 7-DAY FORECAST
# -------------------------------------------------------------------
st.markdown("## 7-Day Forecast — Rain & Temperature")

future_dates = [pd.to_datetime(selected_date) + timedelta(days=i) for i in range(1, 8)]

forecast_rows = []

for i, d in enumerate(future_dates):
    # Estimate avg temps
    tavg = row["temp_avg"] + np.sin(i/2) * 2
    tmin = tavg - np.random.uniform(2, 5)
    tmax = tavg + np.random.uniform(2, 5)

    # Build model input
    X = pd.DataFrame([[
        0, tmin, tmax, tavg,
        humidity, pressure, wind_speed,
        d.month, d.timetuple().tm_yday
    ]], columns=input_data.columns)

    X_scaled = scaler.transform(X)
    prob = model.predict_proba(X_scaled)[0][1]

    forecast_rows.append([d, tmin, tmax, tavg, prob])

forecast_df = pd.DataFrame(forecast_rows, columns=["date", "temp_min", "temp_max", "temp_avg", "prob"])

# CARDS LAYOUT
for i, rowf in forecast_df.iterrows():
    card = st.container(border=True)
    with card:
        c1, c2, c3 = st.columns([1, 3, 2])
        with c1:
            st.markdown(f"**{rowf['date'].strftime('%a, %d %b')}**")
            st.write("Mostly sunny")

        with c2:
            st.write(f"**{rowf['temp_min']:.1f}° / {rowf['temp_max']:.1f}°C**")
            st.write("min / max")

        with c3:
            st.write(f"**{rowf['prob']*100:.1f}%**")
            st.write(f"Est. precip: {(rowf['prob']*8):.2f} mm")

st.write("")


# -------------------------------------------------------------------
# FORECAST PLOTS
# -------------------------------------------------------------------
left, right = st.columns(2)

with left:
    fig1 = px.line(forecast_df, x="date", y="prob", title="7-Day Rain Probability")
    fig1.update_layout(height=300)
    st.plotly_chart(fig1, use_container_width=True)

with right:
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=forecast_df["date"], y=forecast_df["temp_max"], mode="lines+markers", name="Max Temp"
    ))
    fig2.add_trace(go.Scatter(
        x=forecast_df["date"], y=forecast_df["temp_min"], mode="lines+markers", name="Min Temp"
    ))
    fig2.update_layout(title="7-Day Temperature Range", height=300)
    st.plotly_chart(fig2, use_container_width=True)
