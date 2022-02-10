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





new_model = tf.keras.models.load_model('sbdb_my_model')


if __name__=="__main__":

    st.title("**Asteroid Diameter Prediction**")




    start = time.time()


    uploaded_file = st.file_uploader("Choose a CSV file")

    df = pd.read_csv(uploaded_file.name)
    # st.write(os.getcwd())
    st.write("original dataframe")
    st.dataframe(df)

    df.drop(['pha','class','condition_code','spec_B', 'spec_T','per.y','rot_per', 'BV', 'UB','G','data_arc', 'diameter' ],axis=1,inplace=True)

    st.write("Dropping few columns")

    # st.dataframe(df)


    df_predict = df.copy(deep=True)

    index_list = df_predict.index[df_predict['albedo'].isnull()].tolist()

    np.random.seed(0)
    mu = 0.14
    sigma = float(0.01)


    #    print(norm_val)

    for idx in tqdm(index_list):
         df_predict.loc[idx,'albedo']= np.random.normal(mu, sigma ,size=1)
    alblist = df_predict['albedo'].tolist() ###list of created albedo values


    st.write("after inflecting albedo")
    st.dataframe(df_predict)

    
    test_predictions = new_model.predict(df_predict).flatten()

    df_predict['predicted_dia'] = test_predictions.tolist()

    st.dataframe(df_predict)


    # results=search(query)
    # if results is not None:
    #     datafr = pd.DataFrame({'results':results})
    #
    #     st.table(datafr)
    #
    # end = time.time()
    print(start)
