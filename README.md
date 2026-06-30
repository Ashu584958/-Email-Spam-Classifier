# Smart Email Spam Classifier ✉️🤖

An end-to-end Machine Learning web application that uses Natural Language Processing (NLP) to classify text messages or emails into **Spam** or **Ham (Safe)**. 

## 🚀 Features
* **AI Text Preprocessing:** Cleans input using lowercasing, tokenization, stopword removal, and stemming (NLTK PorterStemmer).
* **Feature Extraction:** Converts text data into mathematical weights using `TF-IDF Vectorization`.
* **Classification Intelligence:** Driven by a `Multinomial Naive Bayes` algorithm optimized for fast and precise text prediction.
* **Interactive UI:** A highly responsive dashboard layout built with `Streamlit`.

## 🛠️ Built With
* **Language:** Python 3.13
* **Libraries:** Scikit-Learn, Pandas, Numpy, NLTK
* **Frontend UI:** Streamlit

## 💻 How To Run Locally

1. Clone this repository or download the files.
2. Place your `spam.csv` dataset in the project directory.
3. Train the model to generate the brain files:
   ```bash
   python train_model.py
   ```
4. Run the interactive web interface:
   ```bash
   streamlit run app.py
   ```
