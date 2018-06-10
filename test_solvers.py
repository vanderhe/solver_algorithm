#!/usr/bin/env python3
"""Contains routines to test the solvers module"""

import numpy as np
import solvers

def main():
    """Main testing function."""

    print("\nTest 1")
    aa = np.array([[2.0, 4.0, 4.0], [5.0, 4.0, 2.0], [1.0, 2.0, -1.0]])
    bb = np.array([1.0, 4.0, 2.0])
    xx_expected = np.array([0.666666666666667, 0.416666666666667, -0.5])
    xx_gauss = solvers.gaussian_eliminate(aa, bb)
    _check_result(xx_expected, xx_gauss)

    print("\nTest 2")
    aa = np.array([[2.0, 4.0, 4.0], [1.0, 2.0, -1.0], [5.0, 4.0, 2.0]])
    bb = np.array([1.0, 2.0, 4.0])
    xx_expected = np.array([0.666666666666667, 0.416666666666667, -0.5])
    xx_gauss = solvers.gaussian_eliminate(aa, bb)
    _check_result(xx_expected, xx_gauss)

    print("\nTest 3")
    aa = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    bb = np.array([1.0, 2.0, 3.0])
    xx_expected = None
    xx_gauss = solvers.gaussian_eliminate(aa, bb)
    _check_result(xx_expected, xx_gauss)


def _check_result(expected, obtained):
    """Checks results by printing expected and obtained one."""
    print("Expected:", expected)
    print("Obtained:", obtained)


if __name__ == '__main__':
    main()
