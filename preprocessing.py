import streamlit as st
import re
import pandas as pd
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import swifter

# Define text cleaning function
def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)  # Remove mentions
    text = re.sub(r'#\w+', '', text)  # Remove hashtags
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.lower()  # Convert to lowercase
    return text

# Define stopwords removal function
def stopwords_removal(words, stopwords_set):
    return [word for word in words if word not in stopwords_set]

# Define stemming function
def stemmed_wrapper(term, stemmer):
    return stemmer.stem(term)

def preprocessing():
    st.header("Preprocessing")

    if 'df' in st.session_state and 'stopwords_set' in st.session_state:
        if st.button('Start Preprocessing'):
            df = st.session_state.df

            with st.spinner('Cleaning text...'):
                df['cleaned_text'] = df['full_text'].apply(clean_text)
            st.success('Text cleaning completed.')

            with st.spinner('Tokenizing text...'):
                df['tokenized_text'] = df['cleaned_text'].apply(word_tokenize)
            st.success('Tokenizing completed.')

            with st.spinner('Removing stopwords...'):
                df['stopwords_text'] = df['tokenized_text'].apply(lambda words: stopwords_removal(words, st.session_state.stopwords_set))
            st.success('Stopwords removal completed.')

            with st.spinner('Stemming text...'):
                factory = StemmerFactory()
                stemmer = factory.create_stemmer()
                st.session_state.stemmer = stemmer  # Save the stemmer in session state

                term_dict = {}
                for document in df['stopwords_text']:
                    for term in document:
                        if term not in term_dict:
                            term_dict[term] = stemmed_wrapper(term, stemmer)

                def get_stemmed_term(document):
                    return ' '.join([term_dict[term] for term in document])

                df['stemmed_text'] = df['stopwords_text'].swifter.apply(get_stemmed_term)
            st.success('Stemming completed.')

            st.session_state.df = df
            st.write(df[['full_text', 'cleaned_text', 'tokenized_text', 'stopwords_text', 'stemmed_text']].head())

            st.session_state.step = 2
    else:
        st.warning("Please upload the data and stopword files first.")
