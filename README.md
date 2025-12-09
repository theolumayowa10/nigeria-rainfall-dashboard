# Nigeria Rainfall Prediction Dashboard (ML-Powered)

An interactive, machine-learning–driven rainfall prediction system for Nigeria featuring real-time weather insights, 7-day forecasts, climate analytics, and a modern Streamlit dashboard UI.  
Built by **Mayowa Oluyole**.

---

##  Project Overview

This project provides an end-to-end weather and rainfall prediction solution for Nigeria.  
It combines:

- Synthetic nationwide weather data (2000–2024)
- Machine learning (Random Forest) for next-day rainfall prediction
- Interactive climate visualizations
- A polished Streamlit dashboard with:
  - State-specific weather view  
  - Weekly temperature & rainfall trend  
  - 7-day forecast (with icons, precipitation bars, and summaries)  
  - ML-powered rain-tomorrow probability  
  - Downloadable forecast outputs

This project is designed for **portfolio demonstration**, machine-learning practice, and real-world dashboard development.

---

##  Machine Learning Model

**Model:** RandomForestClassifier  
**Target:** Will it rain tomorrow? (`RainTomorrow`)  
**Features used:**

- rainfall_mm  
- temp_min  
- temp_max  
- temp_avg  
- humidity  
- wind_speed  
- pressure  
- sunshine_hours  
- month  
- dayofyear  

The final trained model (`rainfall_rf_model.pkl`) is loaded in the Streamlit app.

---

##  Dashboard Features

### **1. State Weather Overview**
- Current temperature
- Humidity, pressure, wind speed
- Rain probability
- Weather condition summary

### **2. Weekly Climate Trends**
- Average temperatures  
- Daily rainfall bars  
- Combined dual-axis chart

### **3. ML Rain Prediction**
- User inputs weather variables  
- Model outputs probability of rain tomorrow  
- Scaler applied for preprocessing  

### **4. 7-Day Forecast**
- Daily icons (sunny, cloudy, rainy)  
- Min/max temperature  
- Predicted precipitation  
- Probability bars  
- Summary of weekly rain expectations  

### **5. National Snapshot**
- Monthly rainfall averages  
- Interactive line plots  

---

##  Dataset

Synthetic Nigerian weather dataset (2000–2024):

