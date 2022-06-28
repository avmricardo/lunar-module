from turtle import color
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate
import gravitational as grav
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

initial_moon_pos = np.array([MOON_DIST, 0, 0])  # + initial_earth_pos
initial_moon_vel = np.array([0, MOON_VEL, 0])  # + initial_earth_vel

initial_module_pos = np.array([MODULE_DIST, 0, 0]) + initial_moon_pos
initial_module_vel = np.array([0, MODULE_VEL, 0]) + initial_moon_vel

X = np.concatenate([initial_sun_pos, initial_earth_pos, initial_moon_pos,
                    initial_sun_vel, initial_earth_vel, initial_moon_vel])


Y = np.concatenate([initial_sun_pos, initial_earth_pos, initial_moon_pos, initial_module_pos,
                    initial_sun_vel, initial_earth_vel, initial_moon_vel, initial_module_vel])

# Conditions for a(t, t0, t1, a0)
t0 = 8640
t1 = 8720
a0 = 15
2.4111*DAY

time = 2.42
t = np.arange(0, time*DAY, 0.5)
solution = scipy.integrate.odeint(
    grav.four_body, Y, t, args=(0, EARTH_MASS, MOON_MASS, MODULE_MASS, t0, t1, a0)
)

sun_pos = solution[:, :3]
earth_pos = solution[:, 3:6]
moon_pos = solution[:, 6:9]
module_pos = solution[:, 9: 12]

# Creating figure
fig = plt.figure(figsize=(10, 5))

# Plotting Sun-Earth-Moon system
ax = fig.add_subplot(111, projection="3d")

# ax.plot(sun_pos[:, 0], sun_pos[:, 1], sun_pos[:, 2], color="tab:red")
ax.plot(earth_pos[:, 0], earth_pos[:, 1], earth_pos[:, 2], color="blue")
ax.plot(moon_pos[:, 0], moon_pos[:, 1], moon_pos[:, 2], color='gray')
ax.plot(module_pos[:, 0], module_pos[:, 1], module_pos[:, 2], color='black')

# Plot the final positions of the system
'''
ax.scatter(sun_pos[-1, 0], sun_pos[-1, 1], sun_pos[-1, 2],
            color="tab:red", marker="o", s=100, label="Sun")
'''
ax.scatter(earth_pos[-1, 0], earth_pos[-1, 1], earth_pos[-1, 2],
           color="blue", marker="o", s=100, label="Earth")

ax.scatter(moon_pos[-1, 0], moon_pos[-1, 1], moon_pos[-1, 2],
           color='gray', marker='o', s=100, label="Moon")

ax.scatter(module_pos[-1, 0], module_pos[-1, 1], module_pos[-1, 2],
           color='black', marker='o', s=50, label='Module')


ax.set_xlabel("x (m)", fontsize=14)
ax.set_ylabel("y (m)", fontsize=14)
ax.set_zlabel("z (m)", fontsize=14)
ax.set_xlim(-4e8, 4e8)
ax.set_ylim(-4e8, 4e8)
ax.set_zlim(-1.5e11, 1.5e11)
ax.set_title(f"Earth-Moon-Module System ({time} days)\n", fontsize=14)
plt.legend()
# plt.savefig('sun_earth_moon_module_zoom.jpg')
plt.show()
