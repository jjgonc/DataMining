import unittest
import numpy as np
from scipy.stats import f_oneway
from f_classif import Dataset

class FClassifTests(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.y = np.array([10, 11, 12])
        self.dataset = Dataset()
        self.dataset.set_data(self.X, self.y, ['a', 'b', 'c'], 'd')

    def test_f_classif(self):
        f_values, p_values = self.dataset.f_classif()
        expected_f_values = [np.nan, np.nan, np.nan]
        expected_p_values = [np.nan, np.nan, np.nan]
        self.assertTrue(np.isnan(f_values).all())
        self.assertTrue(np.isnan(p_values).all())

if __name__ == '__main__':
    unittest.main()
