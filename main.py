import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  LabelEncoder
import xgboost as xgb
import numpy as np
st.header("Water Source Point Functionality Prediction App")
df = pd.read_csv("manicaland_dataset.csv")
# load model
best_xgboost_model = xgb.XGBRegressor()
best_xgboost_model.load_model("best_model.json")


st.subheader("Please input or select relevant features about a Water Source Point!")


st.title("Water Pump Prediction")
st.sidebar.title("Input Parameters")
district = st.sidebar.selectbox("District", df["DISTRICT"].unique())
ward = st.sidebar.selectbox("Ward", df["WARD"].unique())
village = st.sidebar.selectbox("Village", df["VILLAGE"].unique())
hh_served = st.sidebar.slider("Households Served", 0, 500, 50)
pump_type = st.sidebar.selectbox("Pump Type", df["PUMP_TYPE"].unique())
outlets = st.sidebar.slider("Number of Outlets", 0, 10, 2)
soak_away_pit = 1 if st.sidebar.checkbox("Soak Away Pit") else 0
vpm_visits = st.sidebar.slider("VPM Visits per Year", 0, 12, 3)
bh_committee = 1 if st.sidebar.checkbox("BH Committee") else 0
seasonality = st.sidebar.selectbox("Seasonality", df["SEASONALITY"].unique())
aquifer_yield = st.sidebar.number_input("Aquifer Yield", min_value=0, value=100)
total_dissolved_solids = st.sidebar.slider("Total Dissolved Solids", 0, 1000, 500)

# Create a button to predict the water pump status
if st.sidebar.button("Predict"):
    # Load the pickled model
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    # Create a dictionary mapping for each categorical variable
    district_map = {district: i for i, district in enumerate(df["DISTRICT"].unique())}
    ward_map = {ward: i for i, ward in enumerate(df["WARD"].unique())}
    village_map = {village: i for i, village in enumerate(df["VILLAGE"].unique())}
    pump_type_map = {pump_type: i for i, pump_type in enumerate(df["PUMP_TYPE"].unique())}
    seasonality_map = {seasonality: i for i, seasonality in enumerate(df["SEASONALITY"].unique())}

    # Create a DataFrame with the input values
    input_data = pd.DataFrame({
        "district": [district],
        "ward": [ward],
        "village": [village],
        "hh_served": [hh_served],
        "pump_type": [pump_type],
        "outlets": [outlets],
        "soak_away_pit": [soak_away_pit],
        "vpm_visits": [vpm_visits],
        "bh_committee": [bh_committee],
        "seasonality": [seasonality],
        "aquifer_yield": [aquifer_yield],
        "total_dissolved_solids": [total_dissolved_solids]
    })

    # Convert the categorical variables to numerical variables using the mapping dictionaries
    input_data["district"] = input_data["district"].apply(lambda x: district_map[x])
    input_data["ward"] = input_data["ward"].apply(lambda x: ward_map[x])
    input_data["village"] = input_data["village"].apply(lambda x: village_map[x])
    input_data["pump_type"] = input_data["pump_type"].apply(lambda x: pump_type_map[x])
    input_data["seasonality"] = input_data["seasonality"].apply(lambda x: seasonality_map[x])

    # Make the prediction
    predictions = best_xgboost_model.predict(input_data)[0]

    # Display the prediction
    st.write("### Prediction")

    # Convert the predictions to the corresponding functionality state labels
    labels = {0: "Fully Functional", 1: "Non-Functional",2:"Partially Functional"}
    functionality_state = [labels[prediction] for prediction in predictions]
    # Output the functionality state
    st.write(f"The water pump is {functionality_state}")



