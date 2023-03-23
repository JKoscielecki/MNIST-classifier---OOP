from DigitClassifier import DigitClassifier
import numpy as np
# Example usage
classifier = DigitClassifier(algorithm="rand")
input_data = np.random.rand(28, 28)
prediction = classifier.predict(input_data)
print("Predicted digit:", prediction)
