import streamlit as st
import pickle

st.title("Twitter Sentiment Analysis")

# Load the trained model
try:
    with open('trained_model.sav', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Model file not found. Please make sure the model file is available.")

# Load the TF-IDF vectorizer
try:
    with open('tfidf_vectorizer.pkl', 'rb') as file:
        tfidf_vectorizer = pickle.load(file)
except FileNotFoundError:
    st.error("TF-IDF vectorizer file not found. Please make sure the vectorizer file is available.")

# Function to get emoji based on sentiment prediction
def get_emoji(prediction):
    if prediction == 0:
        return "😞"  # Sad emoji for negative sentiment
    else:
        return "😊"  # Smiling emoji for positive sentiment

# Input field for the tweet
tweet = st.text_input("Enter your tweet")

# Button to trigger prediction
submit = st.button('Predict')

# Make prediction when button is clicked and text input is not empty
if submit and tweet.strip():  # Check if tweet is not empty
    try:
        # Vectorize the input tweet
        tweet_vectorized = tfidf_vectorizer.transform([tweet])

        # Make prediction
        prediction = model.predict(tweet_vectorized)

        # Display prediction with emoji
        emoji = get_emoji(prediction[0])
        if prediction[0] == 0:
            st.markdown(f"<h1 style='text-align: center;'>{emoji} Negative tweet</h1>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h1 style='text-align: center;'>{emoji} Positive tweet</h1>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"An error occurred: {e}")
elif submit and not tweet.strip():  # Handle case where tweet input is empty
    st.warning("Please enter a tweet.")











