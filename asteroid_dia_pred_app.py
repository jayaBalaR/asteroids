import os

import streamlit as st
import numpy as np
import pandas as pd
import datetime as dt
import json
import time
import csv
import pickle
from io import StringIO
from tqdm import tqdm
import tensorflow as tf

from sklearn.preprocessing import StandardScaler



my_model = tf.keras.models.load_model('my_model')


if __name__=="__main__":

    st.title("**Asteroid Diameter Prediction**")

    start = time.time()

    magnitude = st.slider('magnitude', 3.4, 29.0, 0.1)
    st.write("The H value= ", magnitude, 'au')
    
    albedo = st.slider('albedo', 0.00, 98.00, 0.01)
    st.write("The geometric albedo(ratio) value= ", albedo)
    
    e = st.slider('eccentricity', 0.0, 0.99, 0.1)
    st.write("The eccentricity(ratio) value= ", e)
    
    a = st.slider('semimajor axis', 0.6, 385.0, 0.1)
    st.write("The semimajor axis value= ", a, 'au')
    
    q = st.slider('perihelion distance', 0.08, 40.0, 0.1)
    st.write("The perihelion distance value= ", q, 'au')
    
    i = st.slider('inclination', 0.0, 180.0, 10.0)
    st.write("The inclination value= ", i)
    
    om = st.slider('longitude of ascending node', 0.0, 360.0, 10.0)
    st.write("The longitude of ascending node= ", om)
    
    #'w', 'ma','ad', 'n', 'per', 'moid'
    w = st.slider('argument of perihelion', 0.0, 360.0, 10.0)
    st.write("The peri= ", w)
    
    ma = st.slider('mean anomaly', 0.0, 367.0, 1.0)
    st.write("The mean anomaly= ", ma)
    
    ad = st.slider('aphelion distance', 1.0, 764.0, 1.0)
    st.write("The aphelion distance= ", ad)
    
    n = st.slider('mean motion', 0.0, 1.0, 1.0)
    st.write("The mean motion= ", n)
        
    per = st.slider('orbital period', 181.0, 2.760000e+06, 1.0)
    st.write("The orbital period= ", per)
    
    
    moid = st.slider('earth min orbit dist', 0.0, 39.0, 1.0)
    st.write("earth min orbit dist= ", moid, 'au')

    scaler = StandardScaler()
    
    inp_array = np.array([magnitude,albedo,e,a,q,i,om,w,ma,ad,n,per,moid])
    inp_features = scaler.fit_transform(inp_array)
    st.write(inp_features)
    

    print(start)
