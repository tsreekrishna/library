import unittest
from crop_classification import crop_type_prediction as ctp


class test_ctp(unittest.TestCase):
    def test_data_preprocess(self):
        input_data = {'arg1': 'value1', 'arg2': 123, 'arg3': [1, 2, 3]}
        result = ctp.data_preprocess(input_data)
        self.assertTrue(result)