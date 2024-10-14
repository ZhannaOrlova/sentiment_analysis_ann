from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

class Dataset:
    def __init__(self, reviews, labels):
        self.vectorizer = CountVectorizer()
        self.reviews = reviews
        self.labels = labels

    def preprocess(self):
        X = self.vectorizer.fit_transform([review.text for review in self.reviews]).toarray()
        y = [1 if label == "positive" else 0 for label in self.labels]
        return train_test_split(X, y, test_size=0.2, random_state=42)

