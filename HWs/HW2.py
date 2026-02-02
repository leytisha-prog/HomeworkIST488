
from pkg_resources import get_provider
import streamlit as st
from openai import OpenAI 
import requests
from bs4 import BeautifulSoup
import anthropic


 
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

# LLM Helper

def build_prompt(text: str, summary_type: str, output_language: str) -> str:
    # tune to match Lab 2 "summary type" options
    instructions_by_type = {
        "100-word summary": "Summarize the following document in approximately 100 words.",
        "Two-paragraph summary": "Summarize the following document in two concise paragraphs.",
        "Five bullet points": "Summarize the following document in five bullet points."
    }

    summary_instruction = instructions_by_type.get(
        summary_type, "Summarize the following document."
    )

    return f"""
You are a helpful assistant that summarizes documents based on user instructions.

TASK: {summary_instruction}

OUTPUT LANGUAGE: 
write the summary in {output_language}. Do not include other languages.

STYLE:
- Use clear and concise language for a general audience.
- Maintain the original meaning and key points of the document.
- If the page content is too short to generate the requested summary, indicate that in the response.

WEB PAGE TEXT:
\"\"\"{text[:20000]}\"\"\"
""".strip()

def call_openai(prompt: str, advanced: bool) -> str:
    # OpenAI SDK (python package: openai)
    from openai import OpenAI

    api_key = st.secrets.get("OPENAI_API_KEY", "")
    if not api_key:
        st.error("OpenAI API key is not set in secrets.")
        return "Error: OpenAI API key is missing."

    client = OpenAI(api_key=api_key)

    # Choose OpenAI models 
    model = "gpt-4o" if advanced else "gpt-4o-mini"

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that summarizes web pages based on user instructions."},
            {"role": "user", "content": prompt},
        ]
    )
    return response.choices[0].message.content.strip()

def call_claude(prompt: str, advanced: bool) -> str:
    # Anthropic SDK (python package: anthropic)
    from anthropic import Anthropic

    api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        st.error("Anthropic API key is not set in secrets.")
        return "Error: Anthropic API key is missing."
    
    client = Anthropic(api_key=api_key)
    prompt = str(prompt)

    # Choose Claude models - new models
    model_new = "claude-3-5-sonnet-20240620" if advanced else "claude-3-haiku-20240307"

    # Choose Claude models - legacy models if new models fail 
    model_legacy = "claude-2" if advanced else "claude-1"
   

    response = client.messages.create(
        model=model_new,
        max_tokens=1000,
        messages=[
            {"role": "system", "content": [{"type": "text", "text": prompt}]}],
    
    )
    # Claude responses are returned as content blocks
    return response.choices[0].text.strip()

def validate_key(provider: str) -> None:
    """
    Simple "key validity" check: we make a tiny request to the selected provider.
    If it fails, we raise an error.
    """
    test_prompt = "Say OK."

    if provider == "OpenAI":
        _ = call_openai(test_prompt, advanced=False)
    elif provider == "Claude":
        _ = call_claude(test_prompt, advanced=False)
    else:
        raise ValueError(f"Unknown provider: {provider}")



# User Interface
st.set_page_config(page_title="HW2 – URL Summarizer", page_icon="🌐", layout="wide")
st.title("🌐 HW2 — URL Summarizer with Multiple LLMs")

# URL input at TOP of screen (not sidebar)
url = st.text_input("Enter a web page URL below", placeholder="https://example.com/article")
st.write(
    "Generate a summary of a web page using your choice of LLM provider and model."
)
# Output language dropdown (at least 3 options)
output_language = st.selectbox(
    "Output language",
    ["English", "French", "Spanish", "German", "Chinese", "Japanese", "Portuguese", "Italian","Burmese"],
    index=0,
)
st.write(
    "Select an output language for the summary."
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


use_advanced_model = st.sidebar.checkbox("Use advanced model", value=False)

llm_provider = st.sidebar.selectbox("LLM Provider", ["OpenAI", "Claude"])
st.sidebar.caption("Make sure to set your API keys in Streamlit secrets.")


if summary_type == "100-word summary":
    summary_instruction = "Summarize the following document in approximately 100 words."
elif summary_type == "Two-paragraph summary":
    summary_instruction = "Summarize the following document in two concise paragraphs."
else:
    summary_instruction = "Summarize the following document in five bullet points."

if url:
    document_text = read_url_content(url.strip())

    if document_text and st.button(type="tertiary", label="Generate Summary"):
        with st.spinner("Generating summary..."):
            prompt = (
                f"{summary_instruction}\n\n"
                f"Write the summary in {output_language}.\n\n"
                f"URL:\n{url}\n\n"
                f"Document:\n{document_text}"
            )

            if llm_provider == "OpenAI":
                summary = call_openai(prompt, advanced=use_advanced_model)
            else:
                summary = call_claude(prompt, advanced=use_advanced_model)

        st.subheader("Summary:")
        st.write(summary)
        st.write("_Summary generated successfully!_")
 