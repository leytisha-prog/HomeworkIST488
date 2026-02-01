
import streamlit as st
from openai import OpenAI 
import requests
from bs4 import BeautifulSoup


 
# Use markdown to style generate summary button

st.markdown("""
 <style>
 /* Targets the button element inside a specific Streamlit container */
 div.stButton > button:first-child {
     background-color: #002975; /* Custom background color */
     color: white;              /* Text color */
     border: none;
     border-radius: 8px;        /* Rounded corners */
    padding: 10px 24px;
    cursor: pointer;
    font-size: 16px;
 }
 
 /* Changes style on hover */
 div.stButton > button:first-child:hover {
     background-color: #005fa3;
 }
 
 /* Changes style when the button is active (clicked) */
 div.stButton > button:first-child:active {
     background-color: #003e6b;
 }
 </style>
 """, unsafe_allow_html=True)
 

 # URL Reader (provided by Prof.)
def read_url_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status() # raise an eception for HTTP errors
        soup = BeautifulSoup(response.text, 'html.parser')
        return soup.get_text(separator='\n')
    except requests.RequestException as e:
        st.error(f"Error reading {url}: {e}")
        return None

try:
    client = OpenAI(
        api_key=st.secrets["OPENAI_API_KEY"]
    )
except KeyError:
    st.error("OpenAI API key not found. Please set it in Streamlit secrets.")


# Show title and description.
st.title("Document Summarizer App")
st.write(
    "Upload a PDF or a TXT document below and ask a question about it – GPT will answer! "
)

st.sidebar.header(":blue[Summary Options]")

summary_type = st.sidebar.radio(
    "Choose summary type:",
    [
        "100-word summary",
        "Two-paragraph summary",
        "Five bullet points",
    ],
)

use_advanced_model = st.sidebar.checkbox("Use advanced model (gpt-4o)")

model_name = "gpt-4o-mini" if use_advanced_model else "gpt-4o"

uploaded_file = st.file_uploader(
    "Upload a document (.txt or .pdf)", type=("txt", "pdf")
)

if summary_type == "100-word summary":
    summary_instruction = "Summarize the following document in approximately 100 words."
elif summary_type == "Two-paragraph summary":
    summary_instruction = "Summarize the following document in two concise paragraphs."
else:
    summary_instruction = "Summarize the following document in five bullet points."

if uploaded_file:
    document_text = read_pdf(uploaded_file) if uploaded_file.name.endswith(".pdf") else uploaded_file.read().decode()

    if st.button(type="tertiary", label="Generate Summary"):
        with st.spinner("Generating summary..."):
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that summarizes documents based on user instructions.",
                    },
                    {
                        "role": "user",
                        "content": f"{summary_instruction}\n\nDocument:\n{document_text}",
                    }
                ]
            )
        st.subheader("Summary:")
        st.write(response.choices[0].message.content)
        st.write("_Summary generated successfully!_")

 