import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return 1 / x


def f_d(x):
    return -1 / (x ** 2)


def s1(x):
    if 4 >= x >= 1:
        return a1 + b1 * (x - 1) + c1 * (x - 1) ** 2 + d1 * (x - 1) ** 3
    else:
        return np.nan


def s1_d(x):
    if 4 >= x >= 1:
        return b1 + 2 * c1 * (x - 1) + 3 * d1 * (x - 1) ** 2
    else:
        return np.nan


def s1_dd(x):
    if 4 >= x >= 1:
        return 2 * c1 + 6 * d1 * (x - 1)
    else:
        return np.nan


def s2(x):
    if 4 <= x <= 5:
        return a2 + b2 * (x - 4) + c2 * (x - 4) ** 2 + d2 * (x - 4) ** 3
    else:
        return np.nan


def s2_d(x):
    if 4 <= x <= 5:
        return b2 + 2 * c2*(x - 4) + 3 * d2*(x - 4) ** 2
    else:
        return np.nan


def s2_dd(x):
    if 4 <= x <= 5:
        return 2 * c2 + 6 * d2*(x - 4)
    else:
        return np.nan


a1 = 1
a2 = 1 / 4

A = np.array([[1, 0, 0, 0, 0, 0],
              [3, 9, 27, 0, 0, 0],
              [1, 6, 27, -1, 0, 0],
              [0, 2, 18, 0, -2, 0],
              [0, 0, 0, 1, 1, 1],
              [0, 0, 0, 1, 2, 3]])

b = np.array([-1, 0.25 - a1, 0, 0, 1 / 5 - a2, -1 / 25])

b1, c1, d1, b2, c2, d2 = np.linalg.solve(A, b)

assert np.isclose(s1(1), f(1), atol=1e-3)
assert np.isclose(s1_d(1), f_d(1), atol=1e-3)
assert np.isclose(s1(4), f(4), atol=1e-3)
assert np.isclose(s1_d(4), s2_d(4), atol=1e-3)
assert np.isclose(s1_dd(4), s2_dd(4), atol=1e-3)
assert np.isclose(s2(4), f(4), atol=1e-3)
assert np.isclose(s2(5), f(5), atol=1e-3)
assert np.isclose(s2_d(5), f_d(5), atol=1e-3)

x = np.linspace(1, 5, 500)
plt.plot(x, list(map(f, x)), label=r'$f(x)$')
plt.plot(x, list(map(s1, x)), label=r'$s_1(x)$')
plt.plot(x, list(map(s2, x)), label=r'$s_1(x)$')
plt.title('Cubic spline approximation')
plt.legend()
plt.show()

