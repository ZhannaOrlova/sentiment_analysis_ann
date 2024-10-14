# main.py
from application.train_model import TrainModel
from application.predict_sentiment import PredictSentiment
from core.entities import Review

import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    # Sample dataset
    reviews = [
        Review("I love this movie"),
        Review("This film is terrible"),
        Review("Fantastic movie"),
        Review("Horrible plot"),
    ]
    labels = ["positive", "negative", "positive", "negative"]

    # Train the model
    train_use_case = TrainModel(reviews, labels)
    model, X_test, y_test = train_use_case.execute()

    # Predict sentiment for a new review
    predict_use_case = PredictSentiment(model, train_use_case.dataset.vectorizer)
    new_review = "This movie is amazing"
    prediction = predict_use_case.execute(new_review)
    print(f"Sentiment for '{new_review}' is: {prediction}")

if __name__ == "__main__":
    main()
