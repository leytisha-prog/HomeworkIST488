import pandas as pd 
import streamlit as st

df = pd.read_csv(
    "Hws/news.csv",
    sep=";",
    engine="python",
    quotechar='"',
    on_bad_lines="skip"
)

print(df.head(1).to_dict())


