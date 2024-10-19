import unittest
import numpy as np
from grid import Grid

class TestGrid(unittest.TestCase):

    def test_count(self):
        g = Grid(3, [0, 1], 3)
        self.assertEqual(g.points_count(), 27)

    def test_sample_1d(self):
        g = Grid(1, np.array([0, 1]), 2)
        i = np.array([0, 1], dtype=np.int64)
        self.assertTrue(np.all(g.sample(i) == [
            [0, 1]
        ]))

    def test_sample_2d(self):
        g = Grid(2, np.array([0, 1]), 3)
        i = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int64)
        self.assertTrue(np.all(g.sample(i) == [
            [0, 0, 0, 0.5, 0.5, 0.5, 1, 1, 1],
            [0, 0.5, 1, 0, 0.5, 1, 0, 0.5, 1]
        ]))


if __name__ == '__main__':
    unittest.main()