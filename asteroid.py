import os
import faiss
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


# def search(query):

    # '''
    # search the 5 nearest neighbors
    # '''
    # t=time.time()
    #
    # if len(query)> 0:
    #    query_vector = model.encode([query])
    #    k = 5
    #    top_k = so_index.search(query_vector, k)
    #    st.write("time elapse to retrieve results="+str(time.time()-t))
    #    return [terms[_id] for _id in top_k[1].tolist()[0]]


# def load_index():

    # '''
    # load the index using faiss library
    # '''
    # with open("faiss_index.pkl", "rb") as f:
    #     so_index = faiss.deserialize_index(pickle.load(f))
    # return so_index
new_model = tf.keras.models.load_model("")


if __name__=="__main__":

    st.title("**Asteroid Diameter Prediction**")


    # model = SentenceTransformer('distilbert-base-nli-mean-tokens')

    # encoded_data = pickle.load(open("encoded_data.pkl", 'rb'))


    # dataframe = pd.read_csv(os.getcwd()+'\\results\\fields_altered.csv')


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

    new_model = tf.keras.models.load_model('saved_model/sbdb_my_model')
    # st.write(df_predict.isna().sum())
    # st.write(bytes_data)
    # df_asteroid_updated_null_dataset = pickle.load(open(uploaded_file, 'rb'))
    #
    #
    #
    # st.write(df_asteroid_updated_null_dataset.head())
    # magnitude = st.number_input('Magnitude (between 3.4 to 29)')
    # st.write('The magnitude is ', magnitude)
    #
    # albedo = st.number_input('Albedo (between 0.001 to 1)')
    # st.write('The albedo is ', albedo)
    #
    #
    # eccentricity = st.number_input('Eccentricity (ratio between 0 to 1)')
    # st.write('The eccentricity ratio is ', eccentricity)
    #
    # semimajoraxis = st.number_input('SemiMajorAxis (ratio between 0 to 395 au)')
    # st.write('The semimajoraxis is ', semimajoraxis)
    #
    #
    # perihelion = st.number_input('Perihelion distance ( between 0 to 402 au)')
    # st.write('The semimajoraxis is ', perihelion)
    #
    # inclination = st.number_input('Inclination angle ( between 0 to 180)')
    # st.write('The inclination is ', inclination)
    #
    # om = st.number_input('Longitude of ascending node ( between 0 to 180)')
    # st.write('The longitude of ascending node is ', om)


    # terms = dataframe['concatenatetitletag'].to_list()

    #https://unsplash.com/photos/FZRDvAsgEy8



    # so_index = load_index()

    # query = st.text_input('search text')


    # results=search(query)
    # if results is not None:
    #     datafr = pd.DataFrame({'results':results})
    #
    #     st.table(datafr)
    #
    # end = time.time()
    print(start)
