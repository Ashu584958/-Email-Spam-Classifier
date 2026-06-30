import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Try to download stopwords if they aren't fully cached
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# --- Step 1: Load the Trained ML Files ---
@st.cache_resource
def load_ml_assets():
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('vectorizer.pkl', 'rb') as vec_file:
        vectorizer = pickle.load(vec_file)
    return model, vectorizer

try:
    model, tfidf = load_ml_assets()
except FileNotFoundError:
    st.error("Error: Could not find model.pkl or vectorizer.pkl. Please run train_model.py first!")

# --- Step 2: Text Preprocessing Function ---
ps = PorterStemmer()
all_stopwords = stopwords.words('english')

def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    cleaned_words = [ps.stem(word) for word in words if word not in set(all_stopwords)]
    return ' '.join(cleaned_words)

# --- Step 3: Streamlit User Interface Setup ---
st.set_page_config(page_title="AI Email Spam Guard", page_icon="✉️", layout="centered")

st.title("✉️ Smart Email Spam Classifier")
st.write("Enter the text of an email or message below to check if it is safe or a spam threat.")

# Text Input Area
user_input = st.text_area("Paste message content here:", height=150, placeholder="Type or paste your text here...")

# Predict Button
if st.button("Analyze Message", type="primary"):
    if user_input.strip() == "":
        st.warning("Please enter some text first!")
    else:
        # 1. Clean the input text
        cleaned = clean_text(user_input)
        
        # 2. Transform text to numbers using the loaded vectorizer
        vectorized_text = tfidf.transform([cleaned]).toarray()
        
        # 3. Make prediction
        prediction = model.predict(vectorized_text)[0]
        
        # 4. Show output
        st.subheader("Analysis Result:")
        if prediction == 1:
            st.error("🚨 WARNING: This message is classified as SPAM!")
        else:
            st.success("✅ SAFE: This message is classified as HAM (Legitimate).")
