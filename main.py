import streamlit as st
import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('stopwords')
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
import re

# Load the data
df = pd.read_csv("Ai_QnA.csv", encoding='unicode_escape')
questions_lists = df['Questions'].tolist()
answer_lists = df['Answers'].tolist()

def preprocess(text):
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = nltk.word_tokenize(text.lower())
    tokens = [token for token in tokens if token not in stopwords.words('english')]
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    stemmed_tokens = [stemmer.stem(token) for token in lemmatized_tokens]
    return ' '.join(stemmed_tokens)

def preprocess_with_stopwords(text):
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = nltk.word_tokenize(text.lower())
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    stemmed_tokens = [stemmer.stem(token) for token in lemmatized_tokens]
    return ' '.join(stemmed_tokens)

# Vectorization
vectorizer = TfidfVectorizer(tokenizer=nltk.word_tokenize)
X = vectorizer.fit_transform([preprocess(q) for q in questions_lists])

# Chatbot function
def get_response(text):
    processed_text = preprocess_with_stopwords(text)
    vectorized_text = vectorizer.transform([processed_text])
    similarities = cosine_similarity(vectorized_text, X)
    max_similarity = np.max(similarities)
    if max_similarity > 0.6:
        high_similarity_questions = [q for q, s in zip(questions_lists, similarities[0]) if s > 0.6]

        target_answers = []
        for q in high_similarity_questions:
            q_index = questions_lists.index(q)
            target_answers.append(answer_lists[q_index])

        Z = vectorizer.fit_transform([preprocess_with_stopwords(q) for q in high_similarity_questions])
        processed_text_with_stopwords = preprocess_with_stopwords(text)
        vectorized_text_with_stopwords = vectorizer.transform([processed_text_with_stopwords])
        final_similarities = cosine_similarity(vectorized_text_with_stopwords, Z)
        closest = np.argmax(final_similarities)
        return target_answers[closest]
    else:
        return "I can't answer this question."

# CSS styles
css_styles = """
body {
    background-color: #f0f2f6;
    margin: 0;
    font-family: 'Arial', sans-serif;
}

.container {
    width: 80%;
    max-width: 800px;
    margin: auto;
}

.chat-container {
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
    background-color: #ffffff;
}

.chat-bubble {
    padding: 10px;
    margin: 10px 0;
    border-radius: 10px;
    max-width: 70%;
}

.human-bubble {
    background-color: #007bff;
    color: #ffffff;
    align-self: flex-start;
}

.ai-bubble {
    background-color: #f0f2f6;
    color: #000000;
    align-self: flex-end;
}
"""

# Streamlit app
def main():
    # Include CSS styles
    st.markdown(f"<style>{css_styles}</style>", unsafe_allow_html=True)

    # Create or retrieve the chat history
    chat_history = st.session_state.get("chat_history", [])

    # Create a conversation container
    with st.container():
        st.markdown("<h1 style='text-align: center; color: #007bff;'>Dr. Ai</h1>", unsafe_allow_html=True)
        st.markdown("<hr>", unsafe_allow_html=True)

        # User input form
        user_input = st.text_input("👤 You:")

        if st.button("Send"):
            # Get chatbot response
            response = get_response(user_input)

            # Add user input and AI response to chat history
            chat_history.append(("user", user_input))
            chat_history.append(("ai", response))

            # Update chat history in session state
            st.session_state.chat_history = chat_history

            # Clear the user input by updating its value
            user_input = ""

        # Display chat history
        for speaker, message in chat_history:
            if speaker == "user":
                st.markdown(f'<div class="chat-bubble human-bubble">{message}</div>', unsafe_allow_html=True)
            elif speaker == "ai":
                st.markdown(f'<div class="chat-bubble ai-bubble">{message}</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
