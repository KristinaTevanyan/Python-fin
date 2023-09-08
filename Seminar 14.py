
def test_matrix():
    """
    >>> print(repr(Matrix([[1, 2, 3], [3, 2, 1], [4, 5, 6]])))
    Matrix([[1, 2, 3], [3, 2, 1], [4, 5, 6]])
    >>> print(repr(Matrix([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]) * Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])))
    Matrix([[1, 8, 21, 40], [10, 30, 56, 88], [27, 60, 99, 144]])
    >>> print(repr(Matrix([[1, 2, 3, 4], [4, 5, 6, 7], [8, 9, 10, 11]]) + Matrix([[1, 2, 3, 4], [4, 5, 6, 7], [8, 9, 10, 11]])))
    Matrix([[2, 4, 6, 8], [8, 10, 12, 14], [16, 18, 20, 22]])
    >>> print(repr(Matrix([[1, 2, 3, 4], [4, 5, 6, 7], [8, 9, 10, 11]]) * 2))
    Matrix([[2, 4, 6, 8], [8, 10, 12, 14], [16, 18, 20, 22]])
    >>> Matrix([[1, 2, 3, 4], [4, 5, 6, 7], [8, 9, 10, 11]]) * '2'
    Traceback (most recent call last):
    ...
    homework_14.task02.custom_exceptions.MatrixExc.MatrixTypeError: AN ERROR OCCURRED. MatrixTypeError: 2 is not a 'Matrix' instance
    """
    ...


if __name__ == '__main__':
    import doctest
    doctest.testmod(verbose=True)



NEW_MATRIX_SQR = Matrix([[1, 2, 3], [3, 2, 1], [4, 5, 6]])
NEW_MATRIX_SQR_MUL_TEN_ANS = Matrix([[10, 20, 30], [30, 20, 10], [40, 50, 60]])
NEW_MATRIX_RCT = Matrix([[1, 2, 3, 4], [4, 5, 6, 7], [8, 9, 10, 11]])
NEW_MATRIX_MUL_L = Matrix([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
NEW_MATRIX_MUL_R = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
MATRIX_MUL_ANS = Matrix([[1, 8, 21, 40], [10, 30, 56, 88], [27, 60, 99, 144]])
MATRIX_RCT_SUM_ANS = Matrix([[2, 4, 6, 8], [8, 10, 12, 14], [16, 18, 20, 22]])


@pytest.mark.parametrize('expected, actual', [
    (NEW_MATRIX_SQR, [[1, 2, 3], [3, 2, 1], [4, 5, 6]]),
    (NEW_MATRIX_RCT, [[1, 2, 3, 4], [4, 5, 6, 7], [8, 9, 10, 11]]), ])
def test_create_success(expected, actual):
    assert (expected == Matrix(actual))
    assert (id(expected) != id(Matrix(actual)))


@pytest.mark.parametrize('fail_matrix, expected_exc', [
    ([[1, 2, 3], [3, 2, 1], [4, 5, 'a']], ConsistencyMatrixError),
    ([[1, 2, 3, 4], [4, 5, 6], [8, 9, 10, 11]], ConsistencyMatrixError)])
def test_create_fail_common(fail_matrix, expected_exc):
    with pytest.raises(expected_exc):
        Matrix(fail_matrix)


@pytest.mark.parametrize('fail_matrix, expected_exc', [
    ([[1, 2, 3], [3, 2, 1], [4, 5, 'a']], ConsistencyMatrixError), ])
def test_create_fail_values(fail_matrix, expected_exc):
    with pytest.raises(expected_exc, match="AN ERROR OCCURRED. "
                                           "Inconsistent Matrix Error: "
                                           "All values must be of types 'int' or 'float'"):
        Matrix(fail_matrix)


@pytest.mark.parametrize('fail_matrix, expected_exc', [
    ([[1, 2, 3, 4], [4, 5, 6], [8, 9, 10, 11]], ConsistencyMatrixError)])
def test_create_fail_rows(fail_matrix, expected_exc):
    with pytest.raises(expected_exc, match="AN ERROR OCCURRED. "
                                           "Inconsistent Matrix Error: "
                                           "All rows must be of same length"):
        Matrix(fail_matrix)


@pytest.mark.parametrize('matrix_left, matrix_right, expected', [
    (NEW_MATRIX_MUL_L, NEW_MATRIX_MUL_R, MATRIX_MUL_ANS)
])
def test_matrix_mul_success(matrix_left, matrix_right, expected):
    assert (matrix_left * matrix_right == expected)


@pytest.mark.parametrize('matrix_left, matrix_right, expected_exception', [
    (NEW_MATRIX_MUL_R, NEW_MATRIX_MUL_R, MatrixMultiplyError)
])
def test_matrix_mul_fail(matrix_left, matrix_right, expected_exception):
    with pytest.raises(expected_exception,
                       match="AN ERROR OCCURRED. "
                             "MatrixMultiplyError: "
                             "Operation not permitted if rows amount of first matrix "
                             "is not equal to columns amount of other one"):
        matrix_left * matrix_right


@pytest.mark.parametrize('matrix_left, number, expected', [
    (NEW_MATRIX_SQR, 10, NEW_MATRIX_SQR_MUL_TEN_ANS)
])
def test_matrix_mul_by_num_success(matrix_left, number, expected):
    assert (matrix_left * number == expected)


@pytest.mark.parametrize('matrix_left, number, expected_exception', [
    (NEW_MATRIX_SQR, '10', MatrixTypeError)
])
def test_matrix_mul_by_num_fail(matrix_left, number, expected_exception):
    with pytest.raises(expected_exception, match="AN ERROR OCCURRED. "
                                                 "MatrixTypeError: "
                                                 f"{number} is not a 'Matrix' instance"):
        matrix_left * number


@pytest.mark.parametrize('matrix_left, matrix_right, expected', [
    (NEW_MATRIX_RCT, NEW_MATRIX_RCT, MATRIX_RCT_SUM_ANS)
])
def test_matrix_sum_success(matrix_left, matrix_right, expected):
    assert ((tmp := matrix_left + matrix_right) == expected)
    assert (id(tmp) != id(expected))


@pytest.mark.parametrize('matrix_left, matrix_right, expected_exception', [
    (NEW_MATRIX_RCT, NEW_MATRIX_SQR, MatrixValueError)
])
def test_matrix_sum_fail(matrix_left, matrix_right, expected_exception):
    with pytest.raises(expected_exception,
                       match="AN ERROR OCCURRED. "
                             "MatrixValueError: "
                             "Operation not permitted for different-dimensional matrices"):
        matrix_left + matrix_right


if __name__ == '__main__':
    pytest.main(['-v'])

import unittest



NEW_MATRIX_SQR = Matrix([[1, 2, 3], [3, 2, 1], [4, 5, 6]])
NEW_MATRIX_SQR_MUL_TEN_ANS = Matrix([[10, 20, 30], [30, 20, 10], [40, 50, 60]])
NEW_MATRIX_RCT = Matrix([[1, 2, 3, 4], [4, 5, 6, 7], [8, 9, 10, 11]])
NEW_MATRIX_MUL_L = Matrix([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
NEW_MATRIX_MUL_R = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
MATRIX_MUL_ANS = Matrix([[1, 8, 21, 40], [10, 30, 56, 88], [27, 60, 99, 144]])
MATRIX_RCT_SUM_ANS = Matrix([[2, 4, 6, 8], [8, 10, 12, 14], [16, 18, 20, 22]])


class TestMatrixClass(unittest.TestCase):

    def test_create_success(self):
        assert (NEW_MATRIX_SQR == Matrix([[1, 2, 3], [3, 2, 1], [4, 5, 6]]))

    def test_create_fail_common(self):
        self.assertRaises(ConsistencyMatrixError,
                          Matrix,
                          [[1, 2, 3], [3, 2, 1, 4], [4, 5, 6]])

    def test_eq_success(self):
        self.assertTrue(NEW_MATRIX_SQR == Matrix([[1, 2, 3], [3, 2, 1], [4, 5, 6]]))
        self.assertFalse(NEW_MATRIX_SQR == Matrix([[1, 2, 3], [3, 2, 4], [4, 5, 6]]))

    def test_ne_success(self):
        self.assertFalse(NEW_MATRIX_SQR != Matrix([[1, 2, 3], [3, 2, 1], [4, 5, 6]]))
        self.assertTrue(NEW_MATRIX_SQR != Matrix([[1, 2, 3], [3, 2, 4], [4, 5, 6]]))

    def test_eq_fail(self):
        self.assertRaises(MatrixTypeError,
                          NEW_MATRIX_RCT.__eq__,
                          [[1, 2, 3], [3, 2, 1, 4], [4, 5, 6]])

    def test_ne_fail(self):
        self.assertRaises(MatrixTypeError,
                          NEW_MATRIX_RCT.__ne__,
                          [[1, 2, 3], [3, 2, 1, 4], [4, 5, 6]])

    def test_matrix_mul_success(self):
        self.assertTrue(NEW_MATRIX_MUL_L * NEW_MATRIX_MUL_R == MATRIX_MUL_ANS)

    def test_matrix_mul_fail(self):
        self.assertRaises(MatrixMultiplyError,
                          NEW_MATRIX_MUL_L.__mul__,
                          NEW_MATRIX_SQR)


if __name__ == '__main__':
    unittest.main(verbosity=2)
from .dir_walker import walk_dir
from .common_log_util import common_log

__all__ = [
    'walk_dir'
]
import logging

FORMAT = ("{asctime} - {levelname}: "
          "{msg}")
logging.basicConfig(filename='hw15t01_file_list.txt', filemode='w', format=FORMAT, style='{', level=logging.NOTSET)
common_log = logging.getLogger()


if __name__ == '__main__':
    print('Not for separate use')