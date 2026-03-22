import streamlit as st
from openai import OpenAI



# Default parameters
st.set_page_config(page_title="OpenAI Streamlit App", page_icon=None, layout="centered", initial_sidebar_state="auto", menu_items=None)

    # Configure global settings for the Streamlit app (must be called from the top)
st.set_page_config(
        page_title="HW Manager",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://www.example.com/help',
            'Report a bug': 'https://www.example.com/bug',
            'About': "This is a simple Streamlit app using OpenAI's API."
        }
    )


# Create pages for navigation
HW7_page = st.Page("HWs/HW7.py", title="HW7", icon="📰")
HW1_page = st.Page("HWs/HW1.py", title="HW1", icon="📄")
HW2_page = st.Page("HWs/HW2.py", title="HW2", icon="🧪")
HW3_page = st.Page("HWs/HW3.py", title="HW3", icon="💬")
HW4_page = st.Page("HWs/HW4.py", title="HW4", icon="📚")
HW5_page = st.Page("HWs/HW5.py", title="HW5", icon="📝")


pg = st.navigation([HW7_page, HW2_page, HW3_page, HW4_page, HW5_page, HW1_page])

st.set_page_config(page_title="HW Manager", page_icon=':material/edit:')
pg.run() 
