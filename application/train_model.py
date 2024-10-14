import numpy as np
from infrastructure.neural_network import NeuralNetwork
from infrastructure.dataset import Dataset
from core.entities import Review
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

class TrainModel:
    def __init__(self, reviews, labels):
        self.dataset = Dataset(reviews, labels)

    def execute(self):
        X_train, X_test, y_train, y_test = self.dataset.preprocess()
        input_size = X_train.shape[1]
        model = NeuralNetwork(input_size=input_size, hidden_size=5, output_size=1)
        model.train(X_train, np.array(y_train).reshape(-1, 1))
        return model, X_test, y_test, y_train

    def evaluate(self, model, X_test, y_test):
        # Get predictions from the model
        y_pred = model.predict(X_test)  # Replace with your actual prediction logic
        y_pred = np.array(y_pred)  # Convert y_pred to a NumPy array

        # Binarize predictions
        y_pred = (y_pred > 0.5).astype(int).flatten()

        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, pos_label=1)
        recall = recall_score(y_test, y_pred, pos_label=1)
        f1 = f1_score(y_test, y_pred, pos_label=1)
        conf_matrix = confusion_matrix(y_test, y_pred)

        # Print evaluation metrics
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("Confusion Matrix:")
        print(conf_matrix)

        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['negative', 'positive'], 
                    yticklabels=['negative', 'positive'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()

        return accuracy, precision, recall, f1, conf_matrix
