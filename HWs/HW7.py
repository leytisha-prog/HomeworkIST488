import pandas as pd
import streamlit as st

df = pd.read_csv("news.csv", sep=";")
st.write(df.head())
