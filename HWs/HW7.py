import os
import streamlit as st

#df = pd.read_csv("news.csv", sep=";")
#st.write(df.head())

st.write("Current dircetory:", os.getcwd())
st.write("Files here:", os.listdir())

