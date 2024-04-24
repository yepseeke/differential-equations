import numpy as np
import matplotlib.pyplot as plt

from Dynamicalsystem import DynamicalSystem


def calculate_initial_conditions(x0: float, a: float, b: float, alpha: float, omega: float):
    x10 = alpha * np.sin(x0) + a
    x20 = alpha * np.cos(x0) * x10 + b * omega

    return np.array([x0, x10, x20])


def draw_initial_condition_curve(a: float, b: float, alpha: float, omega: float, x_start: float, x_end, N: int):
    x_conditions = np.linspace(x_start, x_end, N)
    points = []
    for x0 in x_conditions:
        x = calculate_initial_conditions(x0, a, b, alpha, omega)
        points.append(x)

    x = np.array(points)[:, 0]
    y = np.array(points)[:, 1]

    print(np.array(points))
    plt.plot(x, y)


# a, b, alpha, omega = 0, 1, 1, 1
# x0_arr = np.linspace(-15, 15, 12)
def draw_3d_trajectories_on_curve(a: float, b: float, alpha: float, omega: float, x0s):
    ds = DynamicalSystem(f'josephson_{a}_{b}_{alpha}_{omega}')

    ax = plt.figure().add_subplot(projection='3d')
    for x0 in x0s:
        x = calculate_initial_conditions(x0, a, b, alpha, omega)
        print(x)

        ds.plot_3d(ax, x, 400, 10000)


def draw_2d_trajectories_on_curve(a: float, b: float, alpha: float, omega: float, x0s):
    ds = DynamicalSystem(f'josephson_{a}_{b}_{alpha}_{omega}')

    # ax = plt.figure().add_subplot(projection='3d')
    for x0 in x0s:
        x = calculate_initial_conditions(x0, a, b, alpha, omega)
        print(x)

        ds.plot_2d(plt, x, 400, 10000)


def generate_random_points_in_cube(N, L=1):
    x = np.random.uniform(0, L, N)
    y = np.random.uniform(0, L, N)
    z = np.random.uniform(0, L, N)

    points = np.column_stack((x, y, z))
    return points


def draw_3d_random_trajectories(a: float, b: float, alpha: float, omega: float, N, L):
    ds = DynamicalSystem(f'josephson_{a}_{b}_{alpha}_{omega}')

    ax = plt.figure().add_subplot(projection='3d')
    x0s = generate_random_points_in_cube(N, L)
    # print(x0s)
    for x0 in x0s:
        ds.plot_3d(ax, x0, 400, 10000, color='k')


if __name__ == '__main__':
    a, b, alpha, omega = 0, 1, 1, 1
    # x0_arr = np.linspace(-20, 15, 1)
    # draw_2d_trajectories_on_curve(a, b, alpha, omega, x0_arr)
    draw_3d_random_trajectories(a, b, alpha, omega, 10, 6)
    ds = DynamicalSystem(f'josephson_{a}_{b}_{alpha}_{omega}')
    x0 = calculate_initial_conditions(2, a, b, alpha, omega)
    ds.plot_3d(plt, x0, 400, 10000)

    # print(x0)
    # ds.plot_2d(plt, x0, 50, 10000)
    # spect = ds.spectral_density(x0, 50, 5, 8192, 1)
    # plt.plot(spect)

    plt.show()

# a, b, alpha, omega = 0.781, 1, 1, 1
# -15
# 0.13071216
# 0.90069955

# a, b, alpha, omega = 0.785, 1, 1, 1
# -2.5
# 0.18652786
# 0.8505644

# a, b, alpha, omega = 0.783, 1, 1, 1
# 15
# 1.43328784
# -0.08885145

# a, b, alpha, omega = 0.782, 1, 1, 1
# 1.03333333e+01
# -6.61628222e-03
# 1.00406826e+00

# # a, b, alpha, omega = 1, 10, 10, 1 неочевидный предельный цикл
# # a, b, alpha, omega = 0.1, 10, 10, 1 предельный цикл
# # a, b, alpha, omega = 0.1, 100, 10, 1 намотка на тор b >= 11.2
# # a, b, alpha, omega = 0.1, 11.2, 1, 1 еще одна намотка
# # a, b, alpha, omega = 0.1, 0.01, 0.01, 0.06
