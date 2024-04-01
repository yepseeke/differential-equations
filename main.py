import numpy as np
import matplotlib.pyplot as plt

from Dynamicalsystem import DynamicalSystem

if __name__ == '__main__':
    a = 0.0075
    ds = DynamicalSystem(f'rayleigh_9_{a}')
    x0 = np.array([0, 0.99])
    # x0 = np.array([0.11, 0])
    ds.plot_2d(plt, x0, 50, 50000)

    plt.show()
