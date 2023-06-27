import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

data = pd.read_csv('manicaland_dataset.csv')
le = LabelEncoder()

data['DISTRICT'] = le.fit_transform(data['DISTRICT'])
data['VILLAGE'] = le.fit_transform(data['VILLAGE'].astype(str))
data['FUNCTIONAL_STATE'] = le.fit_transform(data['FUNCTIONAL_STATE'].astype(str))
data['SOAK_AWAY_PIT'] = le.fit_transform(data['SOAK_AWAY_PIT'].astype(str))
data['PUMP_TYPE'] = le.fit_transform(data['PUMP_TYPE'].astype(str))
data['PROTECTED'] = le.fit_transform(data['PROTECTED'].astype(str))
data['BH_COMMITTEE'] = le.fit_transform(data['BH_COMMITTEE'].astype(str))
data['SEASONALITY'] = le.fit_transform(data['SEASONALITY'].astype(str))
data['PALATABILITY'] = le.fit_transform(data['PALATABILITY'].astype(str))

data_new = data.drop("DATE OF LAST VISIT", axis=1)
features = ['DISTRICT','WARD','VILLAGE','HH_SERVED','PUMP_TYPE','OUTLETS','SOAK_AWAY_PIT','VPM_VISITS/YEAR',
            'BH_COMMITTEE','SEASONALITY','AQUIFER_YIELD','TOTAL _DISSOLVED -SOLIDS','FUNCTIONAL_STATE']
model_data = data_new[features]

y = model_data['FUNCTIONAL_STATE']
X = model_data.copy()
del X['FUNCTIONAL_STATE']
Predictors = model_data.drop('FUNCTIONAL_STATE',axis=1).columns
feature_name = list(X.columns)
num_feats= 10


X_norm = MinMaxScaler().fit_transform(X)
chi_selector = SelectKBest(chi2, k=num_feats)
chi_selector.fit(X_norm, y)
chi_support = chi_selector.get_support()
chi_feature = X.loc[:,chi_support].columns.tolist()

PredictorScaler=MinMaxScaler()

PredictorScalerFit=PredictorScaler.fit(X)

X=PredictorScalerFit.transform(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

from xgboost import XGBClassifier
clf=XGBClassifier(max_depth=10, learning_rate=0.01, n_estimators=200, objective='binary:logistic', booster='gbtree')

XGB=clf.fit(X_train,y_train)


##FRONTEND UI
st.header("Water Source Point Functionality Prediction App")


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

    predictions=XGB.predict(X_test)(input_data)[0]
    # Display the prediction
    st.write("### Prediction")

    # Convert the predictions to the corresponding functionality state labels
    labels = {0: "Fully Functional", 1: "Non-Functional",2:"Partially Functional"}
    functionality_state = [labels[prediction] for prediction in predictions]
    # Output the functionality state
    st.write(f"The water pump is {functionality_state}")



