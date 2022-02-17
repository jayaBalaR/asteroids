import os
# import faiss
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



placeholder = st.empty()

with placeholder.container():
    username = st.text_input('username')
    passwords = st.text_input('password', type="password", disabled=False)

authorize = 0
if passwords == st.secrets["db_password"]:
    placeholder.empty()
    authorize = 1
else:
    if username is not None and passwords is not None:
        st.error("sorry wrong credentials. retry with right credentials")
        authorize = 0


new_model = tf.keras.models.load_model('sbdb_my_model')

if authorize:
    if __name__=="__main__":
        st.title("**Asteroid Diameter Prediction**")

        start = time.time()

        uploaded_file = st.file_uploader("Choose a CSV file")
    
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file.name)

            st.write("original dataframe")
            st.dataframe(df)

            df.drop(['pha','class','condition_code','spec_B', 'spec_T','per.y','rot_per', 'BV', 'UB','G','data_arc', 'diameter' ],axis=1,inplace=True)

            st.write("Dropping few columns")


            df_predict = df.copy(deep=True)

            index_list = df_predict.index[df_predict['albedo'].isnull()].tolist()

            np.random.seed(0)
            mu = 0.14
            sigma = float(0.01)


            for idx in tqdm(index_list):
                df_predict.loc[idx,'albedo']= np.random.normal(mu, sigma ,size=1)
            alblist = df_predict['albedo'].tolist() ###list of created albedo values


            with st.spinner('Computing predictions'):
                time.sleep(5)
            st.success('Done!')
        

    
            test_predictions = new_model.predict(df_predict).flatten()

            df_predict['predicted_dia'] = test_predictions.tolist()

            st.dataframe(df_predict)



        print(start)
