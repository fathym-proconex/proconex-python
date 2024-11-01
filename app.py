#!/.venv/bin python3
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

from pathlib import Path
from collections import defaultdict
from langchain_openai import AzureChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent


def main():
    # ----------------
    # load environmental variables
    # 
    load_dotenv()
    
# =================================================
# Set-up Streamlit UI
#
    # Title
    st.title("""
             Delta V Event Log Exploration :computer:
             """)
    # Upload file
    uploaded_file = st.file_uploader("Choose a file", type=['csv', '.txt'])

    # Process the file if it’s uploaded
    if uploaded_file is not None:
        # Determine file type and read accordingly
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.txt'):
            df = pd.read_csv(uploaded_file, delimiter='\t', encoding='utf-8')
        else:
            st.warning("Unsupported file type. Please upload a CSV or TXT file.")
            df = None

        # Display DataFrame if loaded successfully
        if df is not None:
            st.divider()
            st.write("Data from the uploaded file:")
            st.dataframe(df)  # Displays the DataFrame in Streamlit
    else:
        st.info("Upload a CSV or TXT file to view its content.")        

    # ===========================================
    # get user input
    # Initialize the key in session state only if it doesn’t exist
    if "user_text_key" not in st.session_state:
        st.session_state["user_text_key"] = ""

    # Now create the text input using the session state key
    user_text = st.text_input("Enter some text", key="user_text_key")

    # Display the current text
    st.write("You entered:", user_text)

    # ======================================
    # Instanciate llm
    llm = AzureChatOpenAI(
        azure_deployment="gpt-4o",
        # api_version="2024-05-01-preview",
        model="gpt-4o",
        temperature=0.0
    )
    
    def call_agent(llm, df):
        agent = create_pandas_dataframe_agent(llm,
                                              df,
                                              agent_type="tool-calling",
                                              verbose=True,
                                              allow_dangerous_code=True)
        return agent
    
    prefix = """import re\nimport pandas as pd\import matplotlib.pyplot as plt\nimport seaborn as sns\n from scipy import stats"""   
    suffix = """Be sure to print the Python code."""
    
    if user_text:
        agent = call_agent(llm, df)
        response = agent.invoke(prefix + user_text + suffix)
        st.divider()
        st.write(response["output"])
        if plt.get_fignums():
            st.pyplot(plt)
        



if __name__ == "__main__":
    main()