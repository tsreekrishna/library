import unittest
import numpy as np
import pandas as pd
from crop_type_prediction import get_croptype_prediction

class TestCropTypePrediction(unittest.TestCase):

    def test_get_croptype_prediction(self):
        # Sample input data
        data = np.random.randint(100, 200, size=(10, 14))
        
        # Test with default parameters
        pred_prob, labels = get_croptype_prediction(data)
        self.assertEqual(len(pred_prob), len(data))
        self.assertEqual(len(labels), len(data))
        
        # Test with XGB algorithm
        pred_prob, labels = get_croptype_prediction(data, algorithm='XGB')
        self.assertEqual(len(pred_prob), len(data))
        self.assertEqual(len(labels), len(data))
        
        # Test with RNN algorithm and batch size
        pred_prob, labels = get_croptype_prediction(data, algorithm='RNN', batch_size=4)
        self.assertEqual(len(pred_prob), len(data))
        self.assertEqual(len(labels), len(data))
        
        # Test with invalid algorithm
        with self.assertRaises(ValueError):
            pred_prob, labels = get_croptype_prediction(data, algorithm='invalid')
            
        # Test with invalid classifier type
        with self.assertRaises(ValueError):
            pred_prob, labels = get_croptype_prediction(data, classifier_type='invalid')
            
if __name__ == '__main__':
    unittest.main()