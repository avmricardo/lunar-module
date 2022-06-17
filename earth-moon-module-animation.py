from turtle import color
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import scipy.integrate
# plt.rcParams.update({'figure.autolayout': True})

# Constants
G = 6.67408e-11  # m^3 / kg*s^2
AU = 1.5e11  # Astronomic unit
EARTH_MASS = 5.9742e24  # Kg mass of the Earth
EARTH_VEL = 29.8E3  # m/s orbital velocity of the Earth around the Sun

SUN_MASS = 1.989e+30  # kg mass of the Sun

MOON_DIST = 3.844e8  # m Distance between the Moon and the Earth
MOON_MASS = 7.36e22
MOON_VEL = 1030  # m/s orbital velocity of the Moon around the Earth

MODULE_MASS = 16400  # Kg mass of the lunar module
# m distance between the Moon and the module plus the radius of the Moon
MODULE_DIST = 1.5e4 + 1.737e6
MODULE_VEL = 1674  # m/s orbital velocity of the lunar module

# Time constants
YEAR = 365*24*60*60
DAY = 24*60*60

# Initial conditions
initial_sun_pos = np.array([0, 0, 0])
initial_sun_vel = np.array([0, 0, 0])

initial_earth_pos = np.array([0.0001, 0, 0])
initial_earth_vel = np.array([0, 0, 0])

initial_moon_pos = np.array([MOON_DIST, 0, 0]) + initial_earth_pos
initial_moon_vel = np.array([0, MOON_VEL, 0]) + initial_earth_vel

initial_module_pos = np.array([MODULE_DIST, 0, 0]) + initial_moon_pos
initial_module_vel = np.array([0, MODULE_VEL, 0]) + initial_moon_vel

X = np.concatenate([initial_sun_pos, initial_earth_pos, initial_moon_pos,
                    initial_sun_vel, initial_earth_vel, initial_moon_vel])

# Time evolution function


def two_body(X, t, m1, m2):
    pos_1 = X[:3]
    pos_2 = X[3:6]
    vel_1 = X[6:9]
    vel_2 = X[9:12]
    r_12 = np.linalg.norm(pos_1 - pos_2)**3
    return np.concatenate(
        [
            vel_1,
            vel_2,
            - G*m2*(pos_1 - pos_2) / r_12,
            - G*m1*(pos_2 - pos_1) / r_12
        ])


def three_body(X, t, m1, m2, m3):
    M = [m1, m2, m3]
    pos_1 = X[:3]
    pos_2 = X[3:6]
    pos_3 = X[6:9]
    pos = [pos_1, pos_2, pos_3]
    vel_1 = X[9:12]
    vel_2 = X[12:15]
    vel_3 = X[15:18]
    r_23 = np.linalg.norm(pos_2 - pos_3)**3
    r_12 = np.linalg.norm(pos_2 - pos_1)**3
    r_13 = np.linalg.norm(pos_3 - pos_1)**3
    acel_1 = sum([-G*m2*(pos_1 - pos_2) / r_12,
                  - G*m3*(pos_1 - pos_3) / r_13])
    acel_2 = sum([-G*m3*(pos_2 - pos_3) / r_23,
                  - G*m1*(pos_2 - pos_1) / r_12])
    acel_3 = sum([-G*m2*(pos_3 - pos_2) / r_23,
                  - G*m1*(pos_3 - pos_1) / r_13])
    return np.concatenate(
        [
            vel_1,
            vel_2,
            vel_3,
            acel_1,
            acel_2,
            acel_3
        ])


def a(t, t0, t1, a0):
    return 0 if t < t0 else a0 if t < t1 else 0


t0 = 0
t1 = 0
a0 = 0


def four_body(X, t, m1, m2, m3, m4):
    # Positions
    pos_1 = X[:3]
    pos_2 = X[3:6]
    pos_3 = X[6:9]
    pos_4 = X[9:12]
    # Velocities
    vel_1 = X[12:15]
    vel_2 = X[15:18]
    vel_3 = X[18:21]
    vel_4 = X[21:24]
    r_12 = np.linalg.norm(pos_1 - pos_2)**3
    r_13 = np.linalg.norm(pos_1 - pos_3)**3
    r_14 = np.linalg.norm(pos_1 - pos_4)**3
    r_23 = np.linalg.norm(pos_2 - pos_3)**3
    r_24 = np.linalg.norm(pos_2 - pos_4)**3
    r_34 = np.linalg.norm(pos_3 - pos_4)**3
    # Acelerations
    acel_1 = -G*m2*(pos_1 - pos_2) / r_12 - G*m3 * \
        (pos_1 - pos_3) / r_13 - G*m4*(pos_1 - pos_4) / r_14
    acel_2 = -G*m1*(pos_2 - pos_1) / r_12 - G*m3 * \
        (pos_2 - pos_3) / r_23 - G*m4*(pos_2 - pos_4) / r_24
    acel_3 = -G*m1*(pos_3 - pos_1) / r_13 - G*m2 * \
        (pos_3 - pos_2) / r_23 - G*m4*(pos_3 - pos_4) / r_34
    acel_4 = -G*m1*(pos_4 - pos_1) / r_14 - G*m2 * \
        (pos_4 - pos_2) / r_24 - G*m3*(pos_4 - pos_3) / r_34
    # if np.linalg.norm(pos_2 - pos_4) < 20000:

    return np.concatenate(
        [
            vel_1,
            vel_2,
            vel_3,
            vel_4,
            np.zeros_like(acel_1),
            acel_2,
            acel_3,
            acel_4 + a(t, t0, t1, a0)*(pos_2 - pos_4) /
            np.linalg.norm(pos_2 - pos_4)
        ]
    )


Y = np.concatenate([initial_sun_pos, initial_earth_pos, initial_moon_pos, initial_module_pos,
                    initial_sun_vel, initial_earth_vel, initial_moon_vel, initial_module_vel])


time = 27
t = np.arange(0, time*DAY, 10)
solution = scipy.integrate.odeint(
    four_body, Y, t, args=(0, EARTH_MASS, MOON_MASS, MODULE_MASS)
)

sun_pos = solution[:, :3]
earth_pos = solution[:, 3:6]
moon_pos = solution[:, 6:9]
module_pos = solution[:, 9: 12]

# Creating figure
fig = plt.figure(figsize=(10, 5))

# Plotting Sun-Earth-Moon system
ax = fig.add_subplot(111, projection="3d")


def plot(frame, body_pos, color):
    points_to_plot = points_per_frame * frame
    ax.plot(
        body_pos[:points_to_plot, 0],
        body_pos[:points_to_plot, 1],
        body_pos[:points_to_plot, 2],
        color=color,
    )


def scatter(frame, body_pos, color, label):
    point_to_plot = points_per_frame * frame
    ax.scatter(
        body_pos[point_to_plot, 0],
        body_pos[point_to_plot, 1],
        body_pos[point_to_plot, 2],
        color=color, marker="o", s=100, label=label,
    )


def update(frame):
    ax.clear()
    positions = (earth_pos, moon_pos, module_pos)
    colors = ("tab:red", "blue", "gray", "black")
    labels = ("Earth", "Moon", "Module")
    for pos, color, label in zip(positions, colors, labels):
        plot(frame, body_pos=pos, color=color)
        scatter(frame, body_pos=pos, color=color, label=label)
    ax.set_xlabel("x (m)", fontsize=14)
    ax.set_ylabel("y (m)", fontsize=14)
    ax.set_zlabel("z (m)", fontsize=14)
    ax.set_xlim(-4e8, 4e8)
    ax.set_ylim(-4e8, 4e8)
    ax.set_zlim(-1.5e11, 1.5e11)
    ax.set_title(f"Earth-Moon-Module System ({time} days)\n", fontsize=14)
    plt.legend()


fps = 60
points = earth_pos.shape[0]
points_per_frame = 7200
hours_ps = 2  # Hours of simulation per second of animation

ani = FuncAnimation(
    fig,
    func=update,
    interval=1000 / fps,
    frames=points // points_per_frame
)

# ani.save("earth_moon_module.gif")
plt.show()
