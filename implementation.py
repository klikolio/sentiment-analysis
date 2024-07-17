import streamlit as st
from preprocessing import clean_text, stopwords_removal, stemmed_wrapper
from nltk.tokenize import word_tokenize

def implementation():
    st.header("Model Implementation")

    if 'svm_model' in st.session_state and 'tfidf_vectorizer' in st.session_state and 'stopwords_set' in st.session_state and 'stemmer' in st.session_state:
        user_input = st.text_area("Input text for sentiment analysis:")

        if st.button("Predict Sentiment"):
            with st.spinner("Predicting..."):
                # Preprocess the input text
                cleaned_text = clean_text(user_input)
                tokenized_text = word_tokenize(cleaned_text)
                stopwords_text = stopwords_removal(tokenized_text, st.session_state.stopwords_set)
                stemmed_text = ' '.join([stemmed_wrapper(term, st.session_state.stemmer) for term in stopwords_text])

                # Vectorize the input text
                input_vector = st.session_state.tfidf_vectorizer.transform([stemmed_text])

                # Predict sentiment
                prediction = st.session_state.svm_model.predict(input_vector)

                # Display the prediction
                st.write(f"Predicted Sentiment: {prediction[0]}")
    else:
        st.warning("Please train the model first.")
