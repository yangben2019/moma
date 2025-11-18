"""
Math module
"""

"""
 #############################################################################
 #                            _     _     _     _                            #
 #                           / \   / \   / \   / \                           #
 #                          ( M ) ( O ) ( M ) ( A )                          #
 #                           \_/   \_/   \_/   \_/                           #
 #                                                                           #
 #               MOMA: Ground vehicles mobility mapping toolkit              #
 #############################################################################

MIT License

Copyright (c) 2024 Dario Sirangelo, Nikolaj Jes Petersen

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import matplotlib.pyplot as plt
import numpy as np

from scipy.interpolate import splprep, splev


def main():

    path = np.array(
        [
            [0, 0],
            [0, 1],
            [0, 2],
            [1, 3],
            [2, 3],
            [3, 2],
            [4, 2],
            [5, 1],
            [6, 1],
            [7, 0],
            [8, 0],
            [9, 0],
            [-4.5, -7],
            [-9, -15],
        ]
    )

    path_mdf = path.copy()
    path_mdf[5, 1] = 3
    path_mdf[8, 1] = 0

    tck, u = splprep([path[:, 0], path[:, 1]], k=3, s=0.0)
    u_fine = np.linspace(0, 1, 100)
    x_smooth, y_smooth = splev(u_fine, tck)

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    ax.plot(path[:, 0] + 0.5, path[:, 1] + 0.5, "r", linewidth=2, label="Path")
    ax.plot(x_smooth + 0.5, y_smooth + 0.5, "b", linewidth=2, label="Smoothened path")

    plt.show()


if __name__ == "__main__":
    main()


import math
import numpy as np
import bisect


class CubicSpline1D:
    def __init__(self, x, y):

        h = np.diff(x)
        if np.any(h < 0):
            raise ValueError("x coordinates must be sorted in ascending order")

        self.a, self.b, self.c, self.d = [], [], [], []
        self.x = x
        self.y = y
        self.nx = len(x)  # dimension of x

        # calc coefficient a
        self.a = [iy for iy in y]

        # calc coefficient c
        A = self.__calc_A(h)
        B = self.__calc_B(h, self.a)
        self.c = np.linalg.solve(A, B)

        # calc spline coefficient b and d
        for i in range(self.nx - 1):
            d = (self.c[i + 1] - self.c[i]) / (3.0 * h[i])
            b = 1.0 / h[i] * (self.a[i + 1] - self.a[i]) - h[i] / 3.0 * (
                2.0 * self.c[i] + self.c[i + 1]
            )
            self.d.append(d)
            self.b.append(b)

    def calc_position(self, x):
        if x < self.x[0]:
            return None
        elif x > self.x[-1]:
            return None

        i = self.__search_index(x)
        dx = x - self.x[i]
        position = (
            self.a[i] + self.b[i] * dx + self.c[i] * dx**2.0 + self.d[i] * dx**3.0
        )

        return position

    def calc_first_derivative(self, x):
        if x < self.x[0]:
            return None
        elif x > self.x[-1]:
            return None

        i = self.__search_index(x)
        dx = x - self.x[i]
        dy = self.b[i] + 2.0 * self.c[i] * dx + 3.0 * self.d[i] * dx**2.0
        return dy

    def calc_second_derivative(self, x):
        if x < self.x[0]:
            return None
        elif x > self.x[-1]:
            return None

        i = self.__search_index(x)
        dx = x - self.x[i]
        ddy = 2.0 * self.c[i] + 6.0 * self.d[i] * dx
        return ddy

    def calc_third_derivative(self, x):
        if x < self.x[0]:
            return None
        elif x > self.x[-1]:
            return None

        i = self.__search_index(x)
        dddy = 6.0 * self.d[i]
        return dddy

    def __search_index(self, x):
        return bisect.bisect(self.x, x) - 1

    def __calc_A(self, h):
        A = np.zeros((self.nx, self.nx))
        A[0, 0] = 1.0
        for i in range(self.nx - 1):
            if i != (self.nx - 2):
                A[i + 1, i + 1] = 2.0 * (h[i] + h[i + 1])
            A[i + 1, i] = h[i]
            A[i, i + 1] = h[i]

        A[0, 1] = 0.0
        A[self.nx - 1, self.nx - 2] = 0.0
        A[self.nx - 1, self.nx - 1] = 1.0
        return A

    def __calc_B(self, h, a):
        B = np.zeros(self.nx)
        for i in range(self.nx - 2):
            B[i + 1] = (
                3.0 * (a[i + 2] - a[i + 1]) / h[i + 1] - 3.0 * (a[i + 1] - a[i]) / h[i]
            )
        return B


class CubicSpline2D:

    def __init__(self, x, y):
        self.s = self.__calc_s(x, y)
        self.sx = CubicSpline1D(self.s, x)
        self.sy = CubicSpline1D(self.s, y)

    def __calc_s(self, x, y):
        dx = np.diff(x)
        dy = np.diff(y)
        self.ds = np.hypot(dx, dy)
        s = [0]
        s.extend(np.cumsum(self.ds))
        return s

    def calc_position(self, s):
        x = self.sx.calc_position(s)
        y = self.sy.calc_position(s)

        return x, y

    def calc_curvature(self, s):
        dx = self.sx.calc_first_derivative(s)
        ddx = self.sx.calc_second_derivative(s)
        dy = self.sy.calc_first_derivative(s)
        ddy = self.sy.calc_second_derivative(s)
        k = (ddy * dx - ddx * dy) / ((dx**2 + dy**2) ** (3 / 2))
        return k

    def calc_curvature_rate(self, s):
        dx = self.sx.calc_first_derivative(s)
        dy = self.sy.calc_first_derivative(s)
        ddx = self.sx.calc_second_derivative(s)
        ddy = self.sy.calc_second_derivative(s)
        dddx = self.sx.calc_third_derivative(s)
        dddy = self.sy.calc_third_derivative(s)
        a = dx * ddy - dy * ddx
        b = dx * dddy - dy * dddx
        c = dx * ddx + dy * ddy
        d = dx * dx + dy * dy
        return (b * d - 3.0 * a * c) / (d * d * d)

    def calc_yaw(self, s):
        dx = self.sx.calc_first_derivative(s)
        dy = self.sy.calc_first_derivative(s)
        yaw = math.atan2(dy, dx)
        return yaw


def calc_spline_course(x, y, ds=0.1):
    sp = CubicSpline2D(x, y)
    s = list(np.arange(0, sp.s[-1], ds))

    rx, ry, ryaw, rk = [], [], [], []
    for i_s in s:
        ix, iy = sp.calc_position(i_s)
        rx.append(ix)
        ry.append(iy)
        ryaw.append(sp.calc_yaw(i_s))
        rk.append(sp.calc_curvature(i_s))

    return rx, ry, ryaw, rk, s


def main_1d():
    import matplotlib.pyplot as plt

    x = np.arange(5)
    y = [1.7, -6, 5, 6.5, 0.0]
    sp = CubicSpline1D(x, y)
    xi = np.linspace(0.0, 5.0)

    plt.plot(x, y, "xb", label="Data points")
    plt.plot(
        xi, [sp.calc_position(x) for x in xi], "r", label="Cubic spline interpolation"
    )
    plt.grid(True)
    plt.legend()
    plt.show()


def main_2d():  # pragma: no cover
    import matplotlib.pyplot as plt

    x = [-2.5, 0.0, 2.5, 5.0, 7.5, 3.0, -1.0]
    y = [0.7, -6, 5, 6.5, 0.0, 5.0, -2.0]
    ds = 0.1  # [m] distance of each interpolated points

    sp = CubicSpline2D(x, y)
    s = np.arange(0, sp.s[-1], ds)

    rx, ry, ryaw, rk = [], [], [], []
    for i_s in s:
        ix, iy = sp.calc_position(i_s)
        rx.append(ix)
        ry.append(iy)
        ryaw.append(sp.calc_yaw(i_s))
        rk.append(sp.calc_curvature(i_s))


if __name__ == "__main__":
    # main_1d()
    # main_2d()
    pass
