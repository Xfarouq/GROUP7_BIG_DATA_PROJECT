import streamlit as st
import dask.dataframe as dd
import pandas as pd
import s3fs
import matplotlib.pyplot as plt
import seaborn as sns
import pydeck as pdk

# --- Page setup ---
st.set_page_config(page_title="NYC Taxi Data Explorer", page_icon="🚖", layout="wide")
st.title("🚖 NYC Taxi Data Explorer")

# --- Load parquet files from S3 using Dask ---
bucket = "nyc-taxi-bigdata-project"
prefix = "processed"
s3_path = f"s3://{bucket}/{prefix}/*.parquet"

# Read directly with Dask
df = dd.read_parquet(s3_path, engine="pyarrow", storage_options={"anon": False})

# --- Data cleaning ---
for col in ["tpep_pickup_datetime", "tpep_dropoff_datetime"]:
    if col in df.columns:
        df[col] = dd.to_datetime(df[col], errors="coerce")

numeric_cols = ["fare_amount", "tip_amount", "trip_distance", "passenger_count"]
for col in numeric_cols:
    if col in df.columns:
        df[col] = df[col].astype(float)

# Trip duration
df["trip_duration"] = (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds() / 60
df = df[(df["trip_duration"] > 0) & (df["trip_duration"] < 180)]

# Compute only after cleaning
df = df.compute()

# --- Basic Visualizations ---
st.header("📊 Basic Visualizations")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("👥 Passenger Count")
    passenger_counts = df["passenger_count"].value_counts().sort_index()
    st.bar_chart(passenger_counts)

with col2:
    st.subheader("💰 Fare Amount")
    st.line_chart(df["fare_amount"].clip(0, 100).sample(1000))

with col3:
    st.subheader("⏱ Trip Duration (mins)")
    st.bar_chart(df["trip_duration"].clip(0, 60).sample(1000))

# --- Advanced Insights ---
st.header("📈 Advanced Insights")

if "fare_amount" in df.columns and "tip_amount" in df.columns:
    st.subheader("💵 Fare vs Tip")
    fig, ax = plt.subplots()
    sns.scatterplot(
        x="fare_amount",
        y="tip_amount",
        data=df.sample(min(2000, len(df))),
        alpha=0.3,
        ax=ax,
    )
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 30)
    st.pyplot(fig)

# Trips by Pickup Hour
st.subheader("🕐 Trips by Pickup Hour")
df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour
hourly_counts = df["pickup_hour"].value_counts().sort_index()
st.bar_chart(hourly_counts)

# --- Geo Visualizations ---
import plotly.express as px

st.subheader("🌍 Geo Visualization")

geo_sample = df.sample(n=min(50000, len(df)), random_state=42)

st.markdown("## Animated Pickup Density Map")

heatmap_data = df.groupby(
    ["pickup_hour", "pickup_longitude", "pickup_latitude"]
).size().reset_index(name="count")

heatmap_sample = heatmap_data.sample(
    n=min(80000, len(heatmap_data)), random_state=42
)

heatmap = px.density_mapbox(
    heatmap_sample,
    lat="pickup_latitude",
    lon="pickup_longitude",
    z="count",
    radius=12,
    zoom=10,
    height=750,
    center={"lat": 40.7128, "lon": -74.0060},
    animation_frame="pickup_hour",
    title="Pickup Density Animation (Hour of Day)"
)

heatmap.update_layout(
    mapbox_style="carto-positron",
    margin={"r":0,"t":40,"l":0,"b":0}
)

st.plotly_chart(heatmap, use_container_width=True)


# --- Model Section ---
import joblib
import numpy as np

st.header("🤖 Model Predictions")

# Load models
try:
    fare_model = joblib.load("best_fare_model.pkl")
    tip_model = joblib.load("best_tip_model.pkl")
    st.success("✅ Models loaded successfully!")
except Exception as e:
    st.error(f"⚠️ Could not load models: {e}")
    fare_model, tip_model = None, None

# Prediction form
st.subheader("📌 Predict Fare & Tip")
trip_distance = st.number_input("Trip Distance (miles)", min_value=0.1, max_value=100.0, value=2.5, step=0.1)
passenger_count = st.number_input("Passenger Count", min_value=1, max_value=6, value=1, step=1)
pickup_hour = st.slider("Pickup Hour (0-23)", min_value=0, max_value=23, value=14)
pickup_day_of_week = st.selectbox("Pickup Day of Week", list(range(7)), format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x])
payment_type = st.selectbox("Payment Type", [1.0, 2.0, 3.0, 4.0], format_func=lambda x: {1.0:"Credit Card",2.0:"Cash",3.0:"No Charge",4.0:"Dispute"}[x])

if st.button("🚖 Predict"):
    if fare_model and tip_model:
        # Rebuild features like during training
        input_data = {
            "trip_distance": [trip_distance],
            "passenger_count": [passenger_count],
            "pickup_hour": [pickup_hour],
            "pickup_day_of_week": [pickup_day_of_week],
            "is_weekend": [1 if pickup_day_of_week >= 5 else 0],
            "payment_type_2.0": [1 if payment_type == 2.0 else 0],
            "payment_type_3.0": [1 if payment_type == 3.0 else 0],
            "payment_type_4.0": [1 if payment_type == 4.0 else 0],
        }

        # Convert to DataFrame
        input_df = pd.DataFrame(input_data)

        # Align with training features (fill missing columns with 0)
        expected_features = fare_model.feature_names_in_
        for col in expected_features:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[expected_features]

        # Predictions
        fare_pred = fare_model.predict(input_df)[0]
        tip_pred = tip_model.predict(input_df)[0]

        st.success(f"💵 Predicted Tip: **${fare_pred:.2f}**")
        st.success(f"💰 Predicted Fare: **${tip_pred:.2f}**")
    else:
        st.warning("⚠️ Models not available for prediction.")