import numpy as np
import matplotlib.pyplot as plt

from Dynamicalsystem import DynamicalSystem


def calculate_initial_conditions(x0: float, a: float, b: float, alpha: float, omega: float):
    x10 = alpha * np.sin(x0) + a
    x20 = alpha * np.cos(x0) * x10 + b * omega

    return np.array([x0, x10, x20])


if __name__ == '__main__':
    a, b, alpha, omega = 1, 10, 10, 1
    ds = DynamicalSystem(f'unknown_{a}_{b}_{alpha}_{omega}')
    x0 = calculate_initial_conditions(0, a, b, alpha, omega)
    print(x0)
    ds.plot_2d(plt, x0, 400, 10000)

    plt.show()
