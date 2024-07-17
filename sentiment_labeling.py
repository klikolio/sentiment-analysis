import streamlit as st
from googletrans import Translator
from nltk.sentiment import SentimentIntensityAnalyzer

# Define translation function with error handling
def translate_text(text, translator):
    if not text:
        return ""
    try:
        translation = translator.translate(text, src='id', dest='en')
        return translation.text.lower()
    except Exception as e:
        return str(e)

# Define sentiment labeling function
def label_sentiment(text, analyzer):
    try:
        sentiment = analyzer.polarity_scores(text)
        if sentiment['compound'] >= 0.05:
            return 'positive'
        elif sentiment['compound'] <= -0.05:
            return 'negative'
        else:
            return 'neutral'
    except Exception as e:
        return str(e)

def sentiment_labeling():
    st.header("Sentiment Labeling")

    if 'df' in st.session_state:
        if st.button('Start Sentiment Labeling'):
            df = st.session_state.df

            with st.spinner('Translating text...'):
                translator = Translator()
                df['translated_text'] = df['stemmed_text'].apply(lambda text: translate_text(text, translator))
            st.success('Text translation completed.')

            with st.spinner('Labeling sentiment...'):
                analyzer = SentimentIntensityAnalyzer()
                df['sentiment'] = df['translated_text'].apply(lambda text: label_sentiment(text, analyzer))
            st.success('Sentiment labeling completed.')

            st.session_state.df = df
            st.write(df[['stemmed_text', 'translated_text', 'sentiment']].head())

            # Count sentiments
            sentiment_counts = df['sentiment'].value_counts()

            # Display sentiment counts
            st.write("\nJumlah masing-masing sentimen:")
            st.write(sentiment_counts)

            # Plot sentiment counts
            st.bar_chart(sentiment_counts)

            st.session_state.step = 3
    else:
        st.warning("Please complete the preprocessing step first.")
