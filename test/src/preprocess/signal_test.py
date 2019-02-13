import unittest
import bmitoolbox as bt
import numpy as np

class Signal(unittest.TestCase):

    def test_epoching(self):
        array = np.array(list(range(100))) * 10
        triggers = [0, 10, 20]
        actual = bt.epoching(raw=array, trig=triggers, size=2)
        expect = np.array([[0, 10], [100, 110], [200, 210]])
        self.assertEqual(expect.size, np.sum(actual == expect))


if __name__ == '__main__':
    unittest.main()
