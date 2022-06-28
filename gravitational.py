# File to write the time evolution functions of the forces between objects

import numpy as np

G = 6.67408e-11  # m^3 / kg*s^2
DAY = 24*60*60  # s


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


def four_body(X, t, m1, m2, m3, m4, t0, t1, a0):
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


def a(t, t0, t1, a0):
    return 0 if t < t0 else a0 if t < t1 else 0
