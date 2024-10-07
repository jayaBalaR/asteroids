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
# from tqdm import tqdm
import tensorflow as tf
import sklearn

from sklearn.preprocessing import StandardScaler



# my_model = tf.keras.models.load_model('my_model')
my_model = keras.saving.load_model("my_model")
scaler = StandardScaler()
placeholder = st.empty()
placeholder2 = st.empty()



if __name__=="__main__":

    st.markdown("**Asteroid Diameter Prediction and PHA classification**")
    
    my_preds = []


    start = time.time()
    
#     def form_callback():
#         st.write(st.session_state.H)
#         st.write(st.session_state.albedo)
#         st.write(st.session_state.e)
#         st.write(st.session_state.a)
    
    with st.form('Form1'):
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            magnitude = st.number_input('magnitude', min_value=3.4, max_value=29.0, key="H")
            st.write("The H value= ", st.session_state.H, 'au')
    
            albedo = st.number_input('albedo', min_value=0.00, max_value=98.00, key="albedo")
            st.write("The geometric albedo(ratio) value= ", st.session_state.albedo)
    
            e = st.number_input('eccentricity', min_value=0.0, max_value=0.99, key="e")
            st.write("The eccentricity(ratio) value= ", st.session_state.e)
    
            a = st.number_input('semimajor axis', min_value=0.6, max_value=385.0, key="a")
            st.write("The semimajor axis value= ", st.session_state.a, 'au')
            
#             st.session_state.albedo
#             st.session_state.H
#             st.session_state.e
#             st.session_state.a
            
        
        with col2:
            q = st.number_input('perihelion distance', min_value=0.08, max_value=40.0, key="q")
            st.write("The perihelion distance value= ", st.session_state.q, 'au')
    
            i = st.number_input('inclination', min_value=0.0, max_value=180.0, key="i")
            st.write("The inclination value= ", st.session_state.i, 'degrees')
    
            om = st.number_input('longitude of ascending node', min_value=0.0, max_value=360.0, key="om")
            st.write("The longitude of ascending node= ", st.session_state.om, 'degrees')
    
            #'w', 'ma','ad', 'n', 'per', 'moid'
            w = st.number_input('argument of perihelion', min_value=0.0, max_value=360.0, key="w")
            st.write("The peri= ", st.session_state.w, 'degrees')
    
            ma = st.number_input('mean anomaly', min_value=0.0, max_value=367.0, key="ma")
            st.write("The mean anomaly= ", st.session_state.ma, 'degrees')
       
        with col3:
        
            ad = st.number_input('aphelion distance', min_value=1.0, max_value=764.0, key="ad")
            st.write("The aphelion distance= ", st.session_state.ad, 'au')
    
            n = st.number_input('mean motion', min_value=0.0, max_value=1.0, key="n")
            st.write("The mean motion= ", st.session_state.n, 'deg/d')
        

            per = st.number_input('orbital period', min_value=181.0, max_value=2.760000e+06, key="per")
            st.write("The orbital period= ", st.session_state.per, 'days')
    
    
            moid = st.number_input('earth min orbit dist', min_value=0.0, max_value=39.0, key="moid")
            st.write("earth min orbit dist= ", st.session_state.moid, 'au')
            
        with col4:        
            inp_array = [[st.session_state.H,st.session_state.albedo,st.session_state.e,st.session_state.a,st.session_state.q,st.session_state.i,st.session_state.om,st.session_state.w,st.session_state.ma,st.session_state.ad,st.session_state.n,st.session_state.per,st.session_state.moid]]
            
            submitted1 = st.form_submit_button('Submit 1')

            val = [[3.4, 0.00, 0.0, 0.6, 0.08, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 181.0, 0.0]]
            my_preds = my_model.predict(inp_array).flatten()
            
            
            if inp_array != val:
                st.success("done computing predictions")
                st.dataframe(my_preds)
                array = [[magnitude, e, a, q, i, om, w, ma, ad, n, per, moid, albedo, my_preds]]
                #array = [[st.session_state.H, st.session_state.albedo, st.session_state.e, st.session_state.a, st.session_state.q, st.session_state.i, st.session_state.om, st.session_state.w, st.session_state.ma, st.session_state.ad, st.session_state.n, st.session_state.per, st.session_state.moid]]
                #st.write(array)


                
                df = pd.DataFrame(array)

                dt_gini = pickle.load(open('gini_sbdbmodel.pkl','rb'), encoding='latin1')

                pha_pred = dt_gini.predict(df)


                if pha_pred[0] == 0:
                    st.markdown("**This is not a PHA**")
                else:
                    st.markdown("**This is  a PHA**")
                end = time.time()
                time_elapsed = end-start
                st.write('total time elapsed since the start is', str(time_elapsed))   
