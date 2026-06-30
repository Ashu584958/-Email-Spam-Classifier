import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import pickle

# --- Step 1: Download Required NLP Assets ---
print("Downloading text cleaning rules...")
nltk.download('stopwords')

# --- Step 2: Load and Clean the Dataset ---
print("Loading dataset...")
# Kaggle's spam.csv uses a specific encoding (latin-1)
df = pd.read_csv('spam.csv', encoding='latin-1')

# Drop unused columns that Kaggle includes by default
df = df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], errors='ignore')

# Rename columns to make them easy to understand
df.columns = ['label', 'text']

# Convert text labels ('spam'/'ham') to binary numbers (1/0)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# --- Step 3: Text Preprocessing Function ---
ps = PorterStemmer()
all_stopwords = stopwords.words('english')

def clean_text(text):
    # Remove everything except standard alphabetical letters
    text = re.sub('[^a-zA-Z]', ' ', text)
    # Convert all text to lowercase
    text = text.lower()
    # Break text into individual words
    words = text.split()
    # Remove stop words (the, is, an) and stem the remaining words (running -> run)
    cleaned_words = [ps.stem(word) for word in words if word not in set(all_stopwords)]
    # Join the words back into a single sentence string
    return ' '.join(cleaned_words)

print("Cleaning email text data (this may take a few seconds)...")
df['cleaned_text'] = df['text'].apply(clean_text)

# --- Step 4: Vectorization (Text to Numbers) ---
print("Converting text to numbers...")
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df['cleaned_text']).toarray()
y = df['label'].values

# --- Step 5: Split Data into Train and Test Sets ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# --- Step 6: Train the Naive Bayes Model ---
print("Training the machine learning model...")
model = MultinomialNB()
model.fit(X_train, y_train)

# --- Step 7: Evaluate the Model Accuracy ---
y_pred = model.predict(X_test)
print("\n=== MODEL TRAINING RESULTS ===")
print(f"Accuracy Score: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Ham (Safe)', 'Spam']))

# --- Step 8: Save Model Files for the Web App ---
print("\nSaving intelligence files to disk...")
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)
with open('vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(tfidf, vectorizer_file)

print("Done! Files saved successfully.")
