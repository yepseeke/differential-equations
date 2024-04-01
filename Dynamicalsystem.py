import numpy as np

from typing import Callable
from numba import jit, njit


class DynamicalSystem:
    def __init__(self, equation_params: str):
        equation_params = equation_params.split(sep='_')

        self.name = equation_params[0]
        self.params = equation_params[1:]
        self.dimension = 0

    def f(self, variables: np.array):

        if self.name == 'lorenz':
            self.dimension = 3

            if variables.shape != (self.dimension,):
                raise Exception(f"Expected {self.dimension}d vector")

            x, y, z = variables[:self.dimension]

            sigma, r, beta = map(float, self.params)

            return np.array([sigma * (y - x), x * (r - z) - y, x * y - beta * z])

        if self.name == 'van-der-pol':
            self.dimension = 2

            if variables.shape != (self.dimension,):
                raise Exception(f"Expected {self.dimension}d vector")

            x, y = variables[:2]

            n, a = map(float, self.params)

            return np.array([y, a * y - x * x * y - x]) if n == 0 \
                else np.array([y, (a * y - x * x * y - x * (1 + n * y)) * (1 + n * y) ** 2])

        if self.name == 'rayleigh':
            self.dimension = 2

            if variables.shape != (self.dimension,):
                raise Exception(f"Expected {self.dimension}d vector")

            x, y = variables[:2]

            n, a = map(float, self.params)

            return np.array([y, (a - y ** 2) * y - x]) if n == 0 \
                else np.array([y, a * y * (1 + n * y) ** 2 - y ** 3 - x * ((1 + n * y) ** 3)])

        else:
            raise ValueError(f"Unknown system: {self.name}")

    def solve(self, x0: np.array, T: float, N: int):
        return rk(self.f, x0, T, N)

    def plot_2d(self, axs, x0: np.array, T: float, N: int):
        solution = self.solve(x0, T, N)
        x = solution[:, 0]
        y = solution[:, 1]

        axs.plot(x, y)


def rk_step(f, x0: np.array, t1: float, t2: float) -> np.array:
    h = t2 - t1

    k1 = h * f(x0)
    k2 = h * f(x0 + 0.5 * k1)
    k3 = h * f(x0 + 0.25 * (k1 + k2))
    k4 = h * f(x0 - k2 + 2 * k3)

    x = x0 + 1.0 / 6.0 * (k1 + 4 * k3 + k4)

    return x


def rk(f, x0: np.array, T: float, N: int) -> np.array:
    points = np.array([x0])
    t = np.linspace(0, T, N + 1)
    for i in range(N):
        if (i + 1) % 100 == 0:
            print(i + 1)
        x = rk_step(f, x0, t[i], t[i + 1])
        points = np.append(points, [x], axis=0)
        x0 = x

    return points
