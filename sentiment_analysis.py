import pandas as pd
import nltk
import re
import sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import classification_report, accuracy_score

# Fix for UnicodeEncodeError on Windows terminal
sys.stdout.reconfigure(encoding='utf-8')

nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

data = {
    "comment": [
        "I love this product, it’s amazing!",
        "This is the worst thing I’ve ever bought.",
        "Great customer service and fast delivery.",
        "Not happy with the quality.",
        "Absolutely fantastic experience!",
        "Horrible. Would not recommend.",
        "Very satisfied, will purchase again.",
        "It was okay, nothing special.",
        "Terrible, completely useless.",
        "The product exceeded my expectations."
    ],
    "sentiment": [1, 0, 1, 0, 1, 0, 1, 0, 0, 1]  # 1 = Positive, 0 = Negative
}

df = pd.DataFrame(data)
ps = PorterStemmer()
corpus = []

for i in range(len(df)):
    comment = re.sub('[^a-zA-Z]', ' ', df['comment'][i])
    comment = comment.lower()
    comment = comment.split()
    comment = [ps.stem(word) for word in comment if word not in stopwords.words('english')]
    comment = ' '.join(comment)
    corpus.append(comment)

cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = BernoulliNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("🔹 Accuracy:", accuracy_score(y_test, y_pred))
print("\n🔹 Classification Report:\n", classification_report(y_test, y_pred))