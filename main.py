import numpy as np
import matplotlib.pyplot as plt

from Dynamicalsystem import DynamicalSystem


def calculate_initial_conditions(x0: float, a: float, b: float, alpha: float, omega: float):
    x10 = alpha * np.sin(x0) + a
    x20 = alpha * np.cos(x0) * x10 + b * omega

    return np.array([x0, x10, x20])


if __name__ == '__main__':
    a, b, alpha, omega = 0.1, 0.01, 0.01, 0.06
    ds = DynamicalSystem(f'josephson_{a}_{b}_{alpha}_{omega}')
    x0 = calculate_initial_conditions(0, a, b, alpha, omega)

    print(x0)
    ds.plot_2d(plt, x0, 2000, 4*8192)
    # spect = ds.spectral_density(x0, 50, 5, 8192, 1)
    # plt.plot(spect)

    plt.show()

# a, b, alpha, omega = 1, 10, 10, 1 неочевидный предельный цикл
# a, b, alpha, omega = 0.1, 10, 10, 1 предельный цикл
# a, b, alpha, omega = 0.1, 100, 10, 1 намотка на тор b >= 11.2
# a, b, alpha, omega = 0.1, 11.2, 1, 1 еще одна намотка
# a, b, alpha, omega = 0.1, 0.01, 0.01, 0.06