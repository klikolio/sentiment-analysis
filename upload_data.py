import streamlit as st
import pandas as pd

def upload_data():
    st.header("Upload Data")

    # Clear existing dataframe
    if 'df' in st.session_state:
        del st.session_state.df
    if 'stopwords_set' in st.session_state:
        del st.session_state.stopwords_set

    # File uploader for CSV and stopwords
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    stopword_file = st.file_uploader("Upload a Stopword file", type="txt")

    if uploaded_file is not None:
        st.session_state.df = pd.read_csv(uploaded_file)
        st.success("CSV file uploaded successfully!")

    if stopword_file is not None:
        txt_stopword = pd.read_csv(stopword_file, names=["stopwords"], header=None)
        st.session_state.stopwords_set = set(txt_stopword["stopwords"][0].split(' '))
        st.success("Stopword file uploaded successfully!")

    if 'df' in st.session_state and 'stopwords_set' in st.session_state:
        st.session_state.step = 1
