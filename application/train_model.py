import numpy as np
from infrastructure.neural_network import NeuralNetwork
from infrastructure.dataset import Dataset
from core.entities import Review

class TrainModel:
    def __init__(self, reviews, labels):
        self.dataset = Dataset(reviews, labels)

    def execute(self):
        X_train, X_test, y_train, y_test = self.dataset.preprocess()
        input_size = X_train.shape[1]
        model = NeuralNetwork(input_size=input_size, hidden_size=5, output_size=1)
        model.train(X_train, np.array(y_train).reshape(-1, 1))
        return model, X_test, y_test
