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
    
    number = st.number_input('Insert a number', min_value=0.0, max_value=10.1, step=0.1)
    st.write('The current number is ', number)

#     magnitude = st.number_input('magnitude', 3.4, 29.0, 0.1)
#     st.write("The H value= ", magnitude, 'au')
    
#     albedo = st.number_input('albedo', 0.00, 98.00, 0.01)
#     st.write("The geometric albedo(ratio) value= ", albedo)
    
#     e = st.number_input('eccentricity', 0.0, 0.99, 0.1)
#     st.write("The eccentricity(ratio) value= ", e)
    
#     a = st.number_input('semimajor axis', 0.6, 385.0, 0.1)
#     st.write("The semimajor axis value= ", a, 'au')
    
#     q = st.number_input('perihelion distance', 0.08, 40.0, 0.1)
#     st.write("The perihelion distance value= ", q, 'au')
    
#     i = st.number_input('inclination', 0.0, 180.0, 10.0)
#     st.write("The inclination value= ", i)
    
#     om = st.number_input('longitude of ascending node', 0.0, 360.0, 10.0)
#     st.write("The longitude of ascending node= ", om)
    
#     #'w', 'ma','ad', 'n', 'per', 'moid'
#     w = st.number_input('argument of perihelion', 0.0, 360.0, 10.0)
#     st.write("The peri= ", w)
    
#     ma = st.number_input('mean anomaly', 0.0, 367.0, 1.0)
#     st.write("The mean anomaly= ", ma)
    
#     ad = st.number_input('aphelion distance', 1.0, 764.0, 1.0)
#     st.write("The aphelion distance= ", ad)
    
#     n = st.number_input('mean motion', 0.0, 1.0, 1.0)
#     st.write("The mean motion= ", n)
        

#     per = st.number_input('orbital period', 181.0, 2.760000e+06, 1.0, 1.0 )
#     st.write("The orbital period= ", per)
    
    
#     moid = st.number_input('earth min orbit dist', 0.0, 39.0, 1.0)
#     st.write("earth min orbit dist= ", moid, 'au')

#     scaler = StandardScaler()

    

#     print(start)
