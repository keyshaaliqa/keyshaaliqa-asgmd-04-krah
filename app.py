import streamlit as st
import pandas as pd
import pickle

st.title("Spaceship Titanic Prediction")
st.write("Predict if passenger is Transported")

model = pickle.load(open("model.pkl", "rb"))
preprocessor = pickle.load(open("preprocessor.pkl", "rb"))

# INPUT FEATURES
HomePlanet = st.selectbox("HomePlanet", ["Earth","Europa","Mars"])
CryoSleep = st.selectbox("CryoSleep", [True, False])
Cabin = st.text_input("Cabin", "B/0/P")
Destination = st.selectbox("Destination",
["TRAPPIST-1e","55 Cancri e","PSO J318.5-22"])
Age = st.number_input("Age", value=30)
VIP = st.selectbox("VIP", [True, False])

RoomService = st.number_input("RoomService", value=0)
FoodCourt = st.number_input("FoodCourt", value=0)
ShoppingMall = st.number_input("ShoppingMall", value=0)
Spa = st.number_input("Spa", value=0)
VRDeck = st.number_input("VRDeck", value=0)

# dummy features tambahan
GroupSize = st.number_input("GroupSize", value=1)
Family = st.number_input("Family", value=0)

if st.button("Predict"):

    data = pd.DataFrame({
        "HomePlanet":[HomePlanet],
        "CryoSleep":[CryoSleep],
        "Cabin":[Cabin],
        "Destination":[Destination],
        "Age":[Age],
        "VIP":[VIP],
        "RoomService":[RoomService],
        "FoodCourt":[FoodCourt],
        "ShoppingMall":[ShoppingMall],
        "Spa":[Spa],
        "VRDeck":[VRDeck],
        "GroupSize":[GroupSize],
        "Family":[Family]
    })

    data_processed = preprocessor.transform(data)

    prediction = model.predict(data_processed)

    st.subheader("Prediction Result")

    if prediction[0] == True:
        st.success("Passenger Transported")
    else:
        st.error("Passenger Not Transported")