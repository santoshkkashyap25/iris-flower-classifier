import unittest
import numpy as np
from model import load_iris_model, predict_iris, get_feature_names
from config import Config

class TestIrisModel(unittest.TestCase):

    def setUp(self):
        # Ensure model is loaded
        load_iris_model(Config.MODEL_PATH)

    def test_feature_names(self):
        features = get_feature_names()
        self.assertEqual(len(features), 4)
        self.assertIn('sepal length (cm)', features)

    def test_predict_setosa(self):
        # Archetypal Setosa sample
        features = [5.1, 3.5, 1.4, 0.2]
        predicted_class, probabilities = predict_iris(features)
        
        self.assertEqual(predicted_class, 0)
        self.assertEqual(len(probabilities), 3)
        self.assertAlmostEqual(sum(probabilities), 1.0, places=4)
        self.assertGreater(probabilities[0], 0.7)

    def test_predict_versicolor(self):
        # Archetypal Versicolor sample
        features = [5.9, 2.8, 4.2, 1.3]
        predicted_class, probabilities = predict_iris(features)
        
        self.assertEqual(predicted_class, 1)
        self.assertEqual(len(probabilities), 3)
        self.assertAlmostEqual(sum(probabilities), 1.0, places=4)
        self.assertGreater(probabilities[1], 0.5)

    def test_predict_virginica(self):
        # Archetypal Virginica sample
        features = [6.5, 3.0, 5.5, 2.0]
        predicted_class, probabilities = predict_iris(features)
        
        self.assertEqual(predicted_class, 2)
        self.assertEqual(len(probabilities), 3)
        self.assertAlmostEqual(sum(probabilities), 1.0, places=4)
        self.assertGreater(probabilities[2], 0.5)

    def test_invalid_feature_length(self):
        predicted_class, probabilities = predict_iris([5.1, 3.5])
        self.assertEqual(predicted_class, -1)
        self.assertEqual(probabilities, [])

    def test_non_numeric_features(self):
        predicted_class, probabilities = predict_iris(["not", "a", "number", "!"])
        self.assertEqual(predicted_class, -1)
        self.assertEqual(probabilities, [])

if __name__ == '__main__':
    unittest.main()
