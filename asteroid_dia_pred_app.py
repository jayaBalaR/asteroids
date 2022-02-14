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





my_model = tf.keras.models.load_model('my_model')


if __name__=="__main__":

    st.title("**Asteroid Diameter Prediction**")

    start = time.time()

    values = st.slider('Select a range of values',0.0, 100.0, (25.0, 75.0))
    st.write('Values:', values)


    print(start)
