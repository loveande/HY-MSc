import unittest
import torch
from dataset import load_data, GraphDataSet
from parse import get_parse
import pickle

class TestDataset(unittest.TestCase):

    def setUp(self):
        self.args = get_parse()

    def test_load_data(self):
        train_loader, test_loader = load_data(self.args)
        
        self.assertIsInstance(train_loader, torch.utils.data.DataLoader)
        self.assertIsInstance(test_loader, torch.utils.data.DataLoader)

    def test_GraphDataSet(self):
        _, _, y_train, y_test = pickle.load(open('dataset.pkl','rb'))
        X_train, X_test = range(len(y_train)), range(len(y_test))

        train_set = GraphDataSet(X_train, y_train)
        test_set = GraphDataSet(X_test, y_test)

        self.assertEqual(len(train_set), len(X_train))
        self.assertEqual(len(test_set), len(X_test))

        index, target = train_set[0]
        self.assertTrue(isinstance(index, int))
        self.assertTrue(isinstance(target, int))

if __name__ == "__main__":
    unittest.main()
