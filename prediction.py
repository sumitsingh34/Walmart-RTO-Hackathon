import streamlit as st
import pickle
import numpy as np
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.point import Point
from geopy.distance import geodesic

model_shelter = pickle.load(open('model_shelter.pkl','rb'))
model_location = pickle.load(open('model_location.pkl','rb'))
data = pd.read_csv("new_data.csv")

#text_input = st.text_input("Enter your address")

def predict_Shelter(longitude,latitude):
    input_value = np.array([[longitude,latitude]]).astype(np.float64)
    prediction = model_shelter.predict(input_value)
    return prediction

def predict_location(longitude,latitude):
    input_value = np.array([[longitude,latitude]]).astype(np.float64)
    prediction = model_location.predict(input_value)
    return prediction


def predict():

    st.title("RTO")
    address = st.text_input("Address")
    #longitude = st.text_input("Longitude")
    #latitude = st.text_input("Latitude")

    geolocator = Nominatim(user_agent="RTO")
    location = geolocator.geocode(address)
    
    if location :
        lat = location.latitude
        lon = location.longitude
    else:
        lat = None
        lon = None
    c = data['shelter_id'].astype('category')
    d = dict(enumerate(c.cat.categories))
    #st.markdown("Go-> {}" .format(d))


    if st.button("Predict Shelter"):
        output = predict_Shelter(lon,lat)
        #res = output.flat[0:output.size]
        res1 = np.array(output).ravel().tolist()
        res = res1[0]
        st.success('The Shelter is -> {}'.format(d.get(res)))
        st.success('Home longitude is {} and latitude is {}'.format(location.longitude,location.latitude))


    if st.button("Predict Location"):
        output = predict_location(lon,lat)
        res1 = np.array(output).ravel().tolist()
        lng1 = res1[0]
        lat1 = res1[1]

        #found_location = geolocator.reverse(Point(lng1,lat1))
        #st.success('Address is {}'.format(found_location.address))
        lost = (lon,lat)
        found = (lng1,lat1)
        
        st.success('The possible longitude {} and latitude {}'.format(lng1,lat1))
        st.success('The distance travelled is {}'.format(geodesic(found, lost).miles))
