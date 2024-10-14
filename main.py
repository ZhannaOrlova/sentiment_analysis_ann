from application.train_model import TrainModel
from application.predict_sentiment import PredictSentiment
from core.entities import Review

import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from transformers import pipeline
from core.entities import Review
from application.train_model import TrainModel
from application.predict_sentiment import PredictSentiment



from transformers import pipeline

def generate_reviews(num_samples: int):
    # Initialize the GPT-2 text generation model
    generator = pipeline('text-generation', model='gpt2')

    # Define prompts for generating positive and negative reviews
    positive_prompt = "Write a positive movie review."
    negative_prompt = "Write a negative movie review."

    reviews = []
    labels = []

    # Generate an equal number of positive and negative reviews
    for _ in range(num_samples // 2):
        pos_review = generator(positive_prompt, max_length=50, num_return_sequences=1)[0]['generated_text']
        reviews.append(Review(pos_review))
        labels.append("positive")

        neg_review = generator(negative_prompt, max_length=50, num_return_sequences=1)[0]['generated_text']
        reviews.append(Review(neg_review))
        labels.append("negative")

    return reviews, labels


def main():
    # Generate a larger dataset with 30 samples
    reviews, labels = generate_reviews(150)

    # Train the model
    train_use_case = TrainModel(reviews, labels)
    model, X_test, y_test, y_train = train_use_case.execute()

    # Predict sentiment for a new review
    predict_use_case = PredictSentiment(model, train_use_case.dataset.vectorizer)
    new_review = "This movie is amazing"
    prediction = predict_use_case.execute(new_review)
    print(f"Sentiment for '{new_review}' is: {prediction}")

    # Evaluate the model
    accuracy, precision, recall, f1, conf_matrix = train_use_case.evaluate(model, X_test, y_test)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Confusion Matrix:\n{conf_matrix}")

if __name__ == "__main__":
    main()
