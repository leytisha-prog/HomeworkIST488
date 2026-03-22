import pandas as pd 
import streamlit as st

df = pd.read_csv(
    "HWs/news.csv",
    sep=";",
    engine="python",
    quotechar='"',
    on_bad_lines="skip"
)




