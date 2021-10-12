# -*- coding: utf-8 -*-

# unit test
import unittest

# context
import context

# original modules
# from context import annin_dofu
# import annin_dofu
from annin_dofu import calc, matrix, parallel, stats, utils

# approximation系クラスのimport
from annin_dofu.response_surface_methodology import Approximation2dFunction as a2f
from annin_dofu.response_surface_methodology import ApproximationRBFInterpolation as ari


class TestMainModule(unittest.TestCase):
    """
    test annin_dofu module
    """

    def setUp(self):
        """
        最初に実行されるメソッド
        """
        print('*' * 80)
        print('test start!!')
        print()

        print('set up test main module')

        # class object
        self.a2f = a2f()
        self.ari = ari()


    def test_func_001(self):
        """
        test method
        """
        import numpy as np
        
        actual = matrix.get_meshgrid_ndarray(3, [1, 2, 3], [1, 3])
        expected = np.array(
            [
                [1, 1, 1],
                [1, 1, 2],
                [1, 1, 3],
                [1, 3, 1],
                [1, 3, 2],
                [1, 3, 3]
            ]
        )

        # self.assertEqual(expected, actual)
        for pair in zip(expected, actual):
            self.assertEqual(pair[0][0], pair[1][0])


    def tearDown(self):
        """
        最後に実行されるメソッド
        """
        print('tear down main module')

        # delete class object
        del self.a2f, self.ari

        print('test end!!')
        print('*' * 80)
        print()


if __name__ == '__main__':
    unittest.main()
