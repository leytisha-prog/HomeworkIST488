import os
import streamlit as st

df = pd.read_csv("HWs/news.csv", sep=";")
st.write(df.head())



