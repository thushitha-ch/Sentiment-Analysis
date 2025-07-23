# 🧠 Sentiment Analysis with Naive Bayes

This project performs *sentiment analysis* on a set of text comments using *Natural Language Processing (NLP)* and the *Bernoulli Naive Bayes* classifier.

It classifies each comment as either:
- *Positive (1)*
- *Negative (0)*

---

## 📁 Project Structure

sentiment_analysis/ ├── sentiment_analysis.py      # Main Python script ├── README.md                  # Project documentation

---

## 📦 Dependencies

Install the required Python packages before running the script:

```bash
pip install pandas scikit-learn nltk

Also, download the NLTK stopwords:

import nltk
nltk.download('stopwords')


---

📊 Dataset Used

A small in-code dataset of 10 comments is used:

"comment": [
    "I love this product, it’s amazing!",
    "This is the worst thing I’ve ever bought.",
    ...
],
"sentiment": [1, 0, ...]  # 1 = Positive, 0 = Negative


---

🛠 Features Implemented

✅ Text cleaning and preprocessing

✅ Stopwords removal using NLTK

✅ Stemming using Porter Stemmer

✅ Feature extraction using CountVectorizer (Bag-of-Words)

✅ Train/test split using train_test_split

✅ Classification using BernoulliNB

✅ Evaluation with accuracy and classification report



---

🚀 How to Run

1. Make sure all libraries are installed.


2. Save the code as sentiment_analysis.py.


3. Run the script:



python sentiment_analysis.py


---

🔍 Output Example

Example output you’ll see in the terminal:

🔹 Accuracy: 1.0

🔹 Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00         1
           1       1.00      1.00      1.00         1

    accuracy                           1.00         2
   macro avg       1.00      1.00      1.00         2
weighted avg       1.00      1.00      1.00         2

> Note: Accuracy may vary because of the small dataset.

👩‍💻 Developed by

Thushitha
