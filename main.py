import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle


# load model
model = xgb.XGBRegressor()
model.load_model("xgb_best_model.json")
#Caching the model for faster loading

df = pd.read_csv('manicaland_dataset.csv')



# Make predictions using the loaded model

##FRONTEND UI
st.header("Manicaland Province Boreholes")



# Add an image to the top sidebar

st.title("Water Pump Functionality")
image = Image.open('borehole.jpg')
st.image(image, width=450)
st.sidebar.title("Water Source Parameters")
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
if st.button("Click Here to Determine the Functionality of the Borehole"):

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

    predictions=model.predict(input_data)[0]
    # Display the prediction
    st.write("### Prediction")

    # Convert the predictions to the corresponding functionality state labels
    labels = {0: "Fully Functional", 1: "Non-Functional",2:"Partially Functional"}
    functionality_state = [labels[prediction] for prediction in predictions]
    # Output the functionality state
    st.write(f"The water pump is {functionality_state}")


