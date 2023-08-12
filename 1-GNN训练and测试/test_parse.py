import unittest
import torch
from parse import get_parse

class TestParse(unittest.TestCase):

    def test_get_parse(self):
        args = get_parse()
        
        self.assertEqual(args.gnn_layers, 2)
        self.assertEqual(args.batch_size, 32)
        self.assertEqual(args.dropout, 0.3)
        self.assertEqual(args.epoch, 100)
        self.assertTrue(isinstance(args.device, torch.device))
        self.assertEqual(args.lr, 1e-5)
        self.assertEqual(args.l2, 1e-5)

if __name__ == "__main__":
    unittest.main()
