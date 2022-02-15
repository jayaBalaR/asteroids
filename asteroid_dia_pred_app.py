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
scaler = StandardScaler()


if __name__=="__main__":

    st.markdown("**Asteroid Diameter Prediction and PHA classification**")

    start = time.time()
    
    with st.container():
        col1, col2, col3 = st.columns(3)
        
        with col1:
            magnitude = st.number_input('magnitude', min_value=3.4, max_value=29.0, step=0.0001)
            st.write("The H value= ", magnitude, 'au')
    
            albedo = st.number_input('albedo', min_value=0.00, max_value=98.00, step=0.0001)
            st.write("The geometric albedo(ratio) value= ", albedo)
    
            e = st.number_input('eccentricity', min_value=0.0, max_value=0.99, step=0.0001)
            st.write("The eccentricity(ratio) value= ", e)
    
            a = st.number_input('semimajor axis', min_value=0.6, max_value=385.0, step=0.0001)
            st.write("The semimajor axis value= ", a, 'au')
        
        with col2:
            q = st.number_input('perihelion distance', min_value=0.08, max_value=40.0, step=0.0001)
            st.write("The perihelion distance value= ", q, 'au')
    
            i = st.number_input('inclination', min_value=0.0, max_value=180.0, step=10.000)
            st.write("The inclination value= ", i)
    
            om = st.number_input('longitude of ascending node', min_value=0.0, max_value=360.0, step=10.000)
            st.write("The longitude of ascending node= ", om)
    
            #'w', 'ma','ad', 'n', 'per', 'moid'
            w = st.number_input('argument of perihelion', min_value=0.0, max_value=360.0, step=10.000)
            st.write("The peri= ", w)
    
            ma = st.number_input('mean anomaly', min_value=0.0, max_value=367.0, step=1.000)
            st.write("The mean anomaly= ", ma)
       
        with col3:
        
            ad = st.number_input('aphelion distance', min_value=1.0, max_value=764.0, step=1.000)
            st.write("The aphelion distance= ", ad)
    
            n = st.number_input('mean motion', min_value=0.0, max_value=1.0, step=0.1)
            st.write("The mean motion= ", n)
        

            per = st.number_input('orbital period', min_value=181.0, max_value=2.760000e+06, step=1.000 )
            st.write("The orbital period= ", per)
    
    
            moid = st.number_input('earth min orbit dist', min_value=0.0, max_value=39.0, step=1.0)
            st.write("earth min orbit dist= ", moid, 'au')

        
        inp_array = [[magnitude,albedo,e,a,q,i,om,w,ma,ad,n,per,moid]]
        st.write(inp_array)
        inp_features = scaler.fit_transform(inp_array)
        st.write(inp_features)
        st.spinner(text="Computing model predictions")
        model_pred = my_model.predict(inp_features).flatten()
        
        st.write("diameter")
        st.table(model_pred)
