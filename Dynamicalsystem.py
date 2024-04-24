import numpy as np

from scipy.fft import fft
from typing import Callable


# from numba import jit, njit


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

        elif self.name == 'van-der-pol':
            self.dimension = 2

            if variables.shape != (self.dimension,):
                raise Exception(f"Expected {self.dimension}d vector")

            x, y = variables[:2]

            n, a = map(float, self.params)

            return np.array([y, a * y - x * x * y - x]) if n == 0 \
                else np.array([y, (a * y - x * x * y - x * (1 + n * y)) * (1 + n * y) ** 2])

        elif self.name == 'rayleigh':
            self.dimension = 2

            if variables.shape != (self.dimension,):
                raise Exception(f"Expected {self.dimension}d vector")

            x, y = variables[:2]

            n, a = map(float, self.params)

            return np.array([y, (a - y ** 2) * y - x]) if n == 0 \
                else np.array([y, a * y * (1 + n * y) ** 2 - y ** 3 - x * ((1 + n * y) ** 3)])
        elif self.name == 'josephson':
            self.dimension = 3

            if variables.shape != (self.dimension,):
                raise Exception(f"Expected {self.dimension}d vector")

            x, y, z = variables[:3]

            a, b, alpha, omega = map(float, self.params)

            return np.array(
                [y, z, alpha * (np.cos(x) * z - np.sin(x) * y ** 2) - omega ** 2 * (y - alpha * np.sin(x) - a)])

        else:
            raise ValueError(f"Unknown system: {self.name}")

    def solve(self, x0: np.array, T: float, N: int):
        return rk(self.f, x0, T, N)

    def plot_2d(self, axs, x0: np.array, T: float, N: int):
        solution = self.solve(x0, T, N)
        x = solution[:, 0]
        y = solution[:, 1]

        axs.plot(x, y)

    def plot_3d(self, axs, x0: np.array, T: float, N: int, color=None):

        solution = self.solve(x0, T, N)
        if color is None:
            axs.plot(solution[:, 0], solution[:, 1], solution[:, 2])
        else:
            axs.plot(solution[:, 0], solution[:, 1], solution[:, 2], color=color)

    # m - amount of periods T
    # N - points in each period
    # coordinate - for what axis calculate spectral density
    # 1 - x
    # 2 - y
    # 3 - z
    def spectral_density(self, x0: np.array, T: float, m: int, N: int, coordinate: int):
        # if self.dimension < coordinate:
        #     raise ValueError(f"No such variable in system: {coordinate}")
        n = m * N
        points = self.solve(x0, m * T, n).T[coordinate - 1]
        print(points)
        res_fft = np.zeros(N)
        for i in range(m):
            curr_points = points[i * N:(i + 1) * N]
            # print(curr_points)
            curr_fft = np.abs(fft(curr_points, N))
            res_fft += np.square(curr_fft)

        return 2 * res_fft[0:N // 2] / (m * T)


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
            print(i + 1, t[i])
        x = rk_step(f, x0, t[i], t[i + 1])
        # print(x)
        points = np.append(points, [x], axis=0)
        x0 = x

    return points
