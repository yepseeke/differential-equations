import numpy as np
import matplotlib.pyplot as plt

from Dynamicalsystem import DynamicalSystem


def calculate_initial_conditions(x0: float, a: float, b: float, alpha: float, omega: float):
    x10 = alpha * np.sin(x0) + a
    x20 = alpha * np.cos(x0) * x10 + b * omega

    return np.array([x0, x10, x20])


def generate_initial_conditions(a: float, b: float, alpha: float, omega: float,
                                start: float, end: float, N: int):
    x0s = np.linspace(start, end, N)
    points = []
    for x0 in x0s:
        points.append(calculate_initial_conditions(x0, a, b, alpha, omega))

    return np.array(points)


def get_initial_conditions(a: float, b: float, alpha: float, omega: float, x0s: np.array):
    points = []
    for x0 in x0s:
        points.append(calculate_initial_conditions(x0, a, b, alpha, omega))

    return np.array(points)


def draw_initial_condition_curve(a: float, b: float, alpha: float, omega: float,
                                 start: float, end, N: int):
    points = generate_initial_conditions(a, b, alpha, omega, start, end, N)

    x = np.array(points)[:, 0]
    y = np.array(points)[:, 1]

    print(np.array(points))
    plt.plot(x, y)


# a, b, alpha, omega = 0, 1, 1, 1
# x0_arr = np.linspace(-15, 15, 12)
def draw_3d_trajectories_on_curve(ax, a: float, b: float, alpha: float, omega: float,
                                  start: float, end, N: int):
    ds = DynamicalSystem(f'josephson_{a}_{b}_{alpha}_{omega}')

    points = generate_initial_conditions(a, b, alpha, omega, start, end, N)

    for x in points:
        ds.plot_3d(ax, x, 200, 50000)


def draw_2d_trajectories_on_curve(a: float, b: float, alpha: float, omega: float, x0s):
    points = get_initial_conditions(a, b, alpha, omega, x0s)
    for x in points:
        ds.plot_3d(plt, x, 400, 10000)


def draw_3d_trajectories_by_points(ax, a: float, b: float, alpha: float, omega: float, points: np.array, color='k'):
    ds = DynamicalSystem(f'josephson_{a}_{b}_{alpha}_{omega}')

    for x in points:
        ds.plot_3d(ax, x, 400, 10000, color=color)


def generate_random_points_in_cube(N, L=1):
    x = np.random.uniform(0, L, N)
    y = np.random.uniform(0, L, N)
    z = np.random.uniform(0, L, N)

    points = np.column_stack((x, y, z))
    return points


def draw_3d_random_trajectories(ax, a: float, b: float, alpha: float, omega: float, N, L):
    ds = DynamicalSystem(f'josephson_{a}_{b}_{alpha}_{omega}')

    points = generate_random_points_in_cube(N, L)
    for x in points:
        ds.plot_3d(ax, x, 400, 10000, color='k')


if __name__ == '__main__':
    ax = plt.figure().add_subplot(projection='3d')

    a, b, alpha, omega = 0, 1, 1, 100

    ds = DynamicalSystem(f'josephson_{a}_{b}_{alpha}_{omega}')

    # points = generate_initial_conditions(a, b, alpha, omega, -1, 1, 1)

    # draw_3d_trajectories_by_points(ax, a, b, alpha, omega, np.array([[0.1, 0.2, -0.2]]))

    draw_3d_trajectories_on_curve(ax, a, b, alpha, omega, -4, 4, 1)

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
