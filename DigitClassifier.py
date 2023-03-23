import numpy as np

class DigitClassificationInterface:
    def predict(self, input_data):
        raise NotImplementedError

class ConvolutionalNeuralNetwork(DigitClassificationInterface):
    def predict(self, input_data):
        # Implement your CNN prediction here
        raise NotImplementedError

class RandomForestClassifier(DigitClassificationInterface):
    def predict(self, input_data):
        # Implement your Random Forest prediction here
        raise NotImplementedError

class RandomModel(DigitClassificationInterface):
    def predict(self, input_data):
        return np.random.randint(0, 10)

class DigitClassifier:
    def __init__(self, algorithm):
        if algorithm == 'cnn':
            self.model = ConvolutionalNeuralNetwork()
        elif algorithm == 'rf':
            self.model = RandomForestClassifier()
        elif algorithm == 'rand':
            self.model = RandomModel()
        else:
            raise ValueError("Invalid algorithm. Choose 'cnn', 'rf', or 'rand'")

    def preprocess_input(self, input_data, model):
        if isinstance(model, ConvolutionalNeuralNetwork):
            return input_data.reshape(1, 28, 28, 1)
        elif isinstance(model, RandomForestClassifier):
            return input_data.reshape(1, -1)
        elif isinstance(model, RandomModel):
            return input_data[9:19, 9:19]

    def predict(self, input_data):
        preprocessed_input = self.preprocess_input(input_data, self.model)
        return self.model.predict(preprocessed_input)
