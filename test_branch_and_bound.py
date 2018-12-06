import os
import unittest
import numpy as np
import argparse

import branch_and_bound as bb

def read_test_case_file(filename):
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, 'test_cases', filename)
    return np.loadtxt(path).astype(np.int32)

def read_test_case(name):
    matrix_file      = '{}_matrix.csv'.format(name)
    constraints_file = '{}_constraints.csv'.format(name)
    reference_file   = '{}_reference.csv'.format(name)
    T = read_test_case_file(matrix_file)
    d = read_test_case_file(constraints_file).flatten()
    x = read_test_case_file(reference_file).flatten()
    return T, d, x[1:], x[0]

class ConvergenceTest(unittest.TestCase):
    def assert_minimize(self, test_case_name):
        T, d, x, w = read_test_case(test_case_name)
        x_my, w_my = bb.minimize(T, d)
        self.assertTrue(np.all( w_my == w ))

    def test_task_1(self): self.assert_minimize('task_1')
    def test_task_2(self): self.assert_minimize('task_2')
    def test_task_3(self): self.assert_minimize('task_3')
    def test_task_6(self): self.assert_minimize('task_6')
    def test_task_8(self): self.assert_minimize('task_8')

if __name__ == '__main__':
    unittest.main()
