import unittest
import numpy as np
from crop_classification.inference import data_preprocess, get_croptype_predictions, get_conformal_predictions

class TestDataPreprocess(unittest.TestCase):

    def test_valid_input(self):
        data = np.random.rand(2,14)
        crop, outliers = data_preprocess(data)

        self.assertEqual(len(crop), 2)
        self.assertEqual(len(outliers), 0)

    def test_invalid_input(self):
        invalid_data = np.array(["a", "b", "c"])
        with self.assertRaises(Exception):
            crop, outliers = data_preprocess(invalid_data)

    def test_outliers(self):
        data = np.random.rand(4, 14)
        crop, outliers = data_preprocess(data)

        self.assertEqual(len(crop), 2)
        self.assertEqual(len(outliers), 2)

class TestGetCropTypePredictions(unittest.TestCase):

    def test_valid_input(self):
        data = np.random.rand(10, 14)
        probs, labels = get_croptype_predictions(data)

        self.assertEqual(probs.shape[0], 10)
        self.assertEqual(len(labels), 10)

    def test_invalid_algorithm(self):
        data = np.random.rand(10, 14)
        with self.assertRaises(Exception):
            probs, labels = get_croptype_predictions(data, algorithm="invalid")

    def test_invalid_classifier(self):
        data = np.random.rand(10, 14)
        with self.assertRaises(Exception):
            probs, labels = get_croptype_predictions(data, classifier_type="invalid")

class TestGetConformalPredictions(unittest.TestCase):

    def test_valid_input(self):
        data = np.random.rand(10, 14)
        sets, labels = get_conformal_predictions(data)
        
        self.assertEqual(len(sets), 10)
        self.assertEqual(len(labels), 10)

    def test_invalid_classifier(self):
        data = np.random.rand(10, 14)
        with self.assertRaises(Exception):
            sets, labels = get_conformal_predictions(data, classifier_type="invalid")

    def test_invalid_alpha(self):
        data = np.random.rand(10, 14)
        with self.assertRaises(Exception):
            sets, labels = get_conformal_predictions(data, alpha=2.0)

if __name__ == '__main__':
    unittest.main()