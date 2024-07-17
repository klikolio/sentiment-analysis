import streamlit as st
from upload_data import upload_data
from preprocessing import preprocessing
from sentiment_labeling import sentiment_labeling
from machine_learning import machine_learning
from implementation import implementation

# Initialize session state variables
if 'step' not in st.session_state:
    st.session_state.step = 0

# Define the Streamlit app
def main():
    st.sidebar.title("Sentiment Analysis")

    # Sidebar buttons for navigation
    st.sidebar.write("Navigation")
    if st.sidebar.button("Upload Data"):
        st.session_state.step = 0
    if st.sidebar.button("Preprocessing"):
        if st.session_state.step >= 1:
            st.session_state.step = 1
        else:
            st.warning("Please complete the 'Upload Data' step first.")
    if st.sidebar.button("Sentiment Labeling"):
        if st.session_state.step >= 2:
            st.session_state.step = 2
        else:
            st.warning("Please complete the 'Preprocessing' step first.")
    if st.sidebar.button("Machine Learning Model"):
        if st.session_state.step >= 3:
            st.session_state.step = 3
        else:
            st.warning("Please complete the 'Sentiment Labeling' step first.")
    if st.sidebar.button("Implementation"):
        if st.session_state.step >= 4:
            st.session_state.step = 4
        else:
            st.warning("Please complete the 'Machine Learning Model' step first.")

    # Step-by-step navigation
    if st.session_state.step == 0:
        upload_data()
    elif st.session_state.step == 1:
        preprocessing()
    elif st.session_state.step == 2:
        sentiment_labeling()
    elif st.session_state.step == 3:
        machine_learning()
    elif st.session_state.step == 4:
        implementation()

if __name__ == "__main__":
    main()
