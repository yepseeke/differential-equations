import numpy as np
import matplotlib.pyplot as plt

from Dynamicalsystem import DynamicalSystem

if __name__ == '__main__':
    a, b, alpha, omega = 0.1, 0, 0.1, 10
    ds = DynamicalSystem(f'unknown_{a}_{b}_{alpha}_{omega}')
    x0 = np.array([0, 0.99, 0])
    # x0 = np.array([0.11, 0])
    ds.plot_2d(plt, x0, 100, 10000)

    plt.show()
