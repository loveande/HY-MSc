import unittest
import numpy as np
from process_data import cosine_similarity

class TestProcessData(unittest.TestCase):

    def test_cosine_similarity(self):
        vector = np.array([1, 2, 3])
        matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        similarity = cosine_similarity(vector, matrix)

        self.assertIsInstance(similarity, np.ndarray)
        
        self.assertEqual(len(similarity), matrix.shape[0])

        for value in similarity:
            self.assertTrue(-1 <= value <= 1)

        self.assertAlmostEqual(similarity[0], 1, places=6)

if __name__ == "__main__":
    unittest.main()
