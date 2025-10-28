import streamlit as st
import numpy as np
import joblib
from datetime import date

# --- Load Models and Encoders ---
le_city = joblib.load('models/le_city.pkl')
le_county = joblib.load('models/le_county.pkl')
le_fuel = joblib.load('models/le_fuel.pkl')

scaler_reg = joblib.load('models/scaler_regression.pkl')
lr_model = joblib.load('models/linear_regression.pkl')

scaler_clf = joblib.load('models/scaler_classification.pkl')
rf_model = joblib.load('models/random_forest.pkl')
ada_model = joblib.load('models/adaboost.pkl')
gb_model = joblib.load('models/gradient_boosting.pkl')

kmeans = joblib.load('models/kmeans.pkl')
scaler_cluster = joblib.load('models/scaler_clustering.pkl')

cities = joblib.load('models/cities.pkl')
counties = joblib.load('models/counties.pkl')
median_price = joblib.load('models/median_price.pkl')

# --- UI Layout ---
st.title("⛽ Fuel Price Forecasting - UK 🇬🇧")
st.write("Predict and analyze petrol/diesel price trends using trained ML models.")

city = st.selectbox("Select City", cities)
county = st.selectbox("Select County", counties)
fuel_type = st.selectbox("Select Fuel Type", ["Petrol", "Diesel"])

today = date.today()
month = today.month
day = today.day
dayofweek = today.weekday()

st.write("Date (Auto Detected):", today)

rolling_mean = st.number_input("Average of Last 3 Days (Rolling Mean)", value=150.0)
rolling_std = st.number_input("Standard Deviation (Rolling Std)", value=1.0)

if st.button("🔮 Predict Fuel Price"):
    # Encode inputs
    city_enc = le_city.transform([city])[0]
    county_enc = le_county.transform([county])[0]
    fuel_enc = le_fuel.transform([fuel_type])[0]

    features = np.array([[city_enc, county_enc, fuel_enc, month, day, dayofweek, rolling_mean, rolling_std]])

    # --- Regression ---
    reg_scaled = scaler_reg.transform(features)
    predicted_price = lr_model.predict(reg_scaled)[0]

    # --- Classification ---
    clf_scaled = scaler_clf.transform(features)
    gb_pred = gb_model.predict(clf_scaled)[0]
    category = "High Price" if gb_pred == 1 else "Low Price"

    # --- Clustering ---
    cluster_scaled = scaler_cluster.transform([[predicted_price, city_enc, county_enc, fuel_enc, month]])
    cluster = kmeans.predict(cluster_scaled)[0]

    # --- Results ---
    st.success(f"Predicted Price: £{predicted_price:.2f}")
    st.info(f"Category: {category}")
    st.write(f"Cluster Group: {cluster} (0=Low, 1=Medium, 2=High Trend)")

    if predicted_price > median_price:
        st.warning(f"Price is above the median (£{median_price:.2f})")
    else:
        st.success(f"Price is below the median (£{median_price:.2f})")
