from core.entities import Review, Sentiment

class PredictSentiment:
    def __init__(self, model, vectorizer):
        self.model = model
        self.vectorizer = vectorizer

    def execute(self, review_text):
        review = Review(review_text)
        vectorized = self.vectorizer.transform([review.text]).toarray()
        prediction = self.model.predict(vectorized)[0]
        return Sentiment.POSITIVE if prediction == 1 else Sentiment.NEGATIVE
