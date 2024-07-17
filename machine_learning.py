import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

def machine_learning():
    st.header("Machine Learning Model")

    if 'df' in st.session_state:
        if st.button('Start Machine Learning'):
            df = st.session_state.df

            with st.spinner('Creating TF-IDF vectors...'):
                tfidf_vectorizer = TfidfVectorizer()
                X = tfidf_vectorizer.fit_transform(df['stemmed_text'])
                y = df['sentiment']
            st.success('TF-IDF vector creation completed.')

            with st.spinner('Splitting data...'):
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            st.success('Data splitting completed.')

            with st.spinner('Training SVM model...'):
                svm_model = SVC(kernel='linear')
                svm_model.fit(X_train, y_train)
            st.success('SVM model training completed.')

            with st.spinner('Predicting labels...'):
                y_pred = svm_model.predict(X_test)
            st.success('Label prediction completed.')

            with st.spinner('Evaluating model...'):
                unique_classes = y.unique()
                target_names = unique_classes.tolist()
                accuracy = accuracy_score(y_test, y_pred)
                classification_report_str = classification_report(y_test, y_pred, target_names=target_names, labels=unique_classes)
                conf_matrix = confusion_matrix(y_test, y_pred, labels=unique_classes)
                precision = precision_score(y_test, y_pred, average='weighted', labels=unique_classes)
                recall = recall_score(y_test, y_pred, average='weighted', labels=unique_classes)
                f1 = f1_score(y_test, y_pred, average='weighted', labels=unique_classes)
            st.success('Model evaluation completed.')

            # Displaying evaluation metrics
            st.write("Classification report:")
            st.text(classification_report_str)
    

            # Plotting confusion matrix
            st.write("Confusion Matrix:")
            fig, ax = plt.subplots()
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names, ax=ax)
            ax.set_xlabel('Prediksi Label')
            ax.set_ylabel('Label Sebenarnya')
            ax.set_title('Confusion Matrix')
            st.pyplot(fig)

            # Save the model and vectorizer in session state
            st.session_state.svm_model = svm_model
            st.session_state.tfidf_vectorizer = tfidf_vectorizer

            st.session_state.step = 4
    else:
        st.warning("Please complete the sentiment labeling step first.")
