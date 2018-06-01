# -*- coding: utf-8 -*-
"""
Created on Tue May 29 19:22:38 2018

@author: Anant
"""

import random
import math
import numpy as np
import matplotlib.pyplot as plt
from sgp4.earth_gravity import wgs72
from sgp4.io import twoline2rv

line1 = ('1 00005U 58002B   00179.78495062  ''.00000023  00000-0  28098-4 0  4753')
line2 = ('2 00005  34.2682 348.7242 1859667 ''331.7664  19.3264 10.82419157413667')


satellite = twoline2rv(line1, line2, wgs72)

year = 2020
month = 11
day = 27
hour = 10
minute = 23
second = 57

plot_y = []
plot_x = []

q = [0, 0, 0, 0]

for i in range(0, 4):
    q[i] = random.uniform(0, 1)  # initial quaternion values between 0 and 1

# print (q[0],q[1],q[2],q[3])

# 4x4 matrix
X = [[q[0], -q[1], -q[2], -q[3]],
     [q[1], q[0], -q[3], q[1]],
     [q[2], q[3], q[0], -q[1]],
     [q[3], -q[2], q[1], q[0]]]
# 4x1 matrix
Y = [0, 0.34906585, 0.34906585, 0.349065850]
# print(X,Y)

# rate_of_change is 4x1
rate_of_change = [0, 0, 0, 0]

for i in range(0, 4):
    for j in range(0, 4):
        rate_of_change[i] += X[i][j] * Y[j]

print("Initial rate of change: ")
for r in rate_of_change:
    print(r)

for delta_t in range(0, 1000, 100):
    print("Quaternion after ", delta_t, "milliseconds is: "),
    for i in range(0, 4):
        q[i] = q[i] + (rate_of_change[i] * delta_t * 0.001)
    mod_q = math.sqrt(q[0] ** 2 + q[1] ** 2 + q[2] ** 2 + q[3] ** 2)
    for i in range(0, 4):
        q[i] = q[i] / mod_q
        print(" ", q[i], " "),
    print("")
    q_star = [q[0], -q[1], -q[2], -q[3]]
    X = [[q[0], -q[1], -q[2], -q[3]],
         [q[1], q[0], -q[3], q[1]],
         [q[2], q[3], q[0], -q[1]],
         [q[3], -q[2], q[1], q[0]]]

    JD = 367 * year - int(7 * (year + int((month + 9) / 12))) + int(275 * month / 9) + day + 1721013.5 + (hour / 24) + (
            minute / 14400) + (second / 86400)
    T_UT1 = (JD - 2451545.0) / 36525
    lambda_Msun = 280.4606184 + (36000.77005361 * T_UT1)
    M_Sun = 357.5277233 + (35999.05034 * T_UT1)
    lambda_elliptic = M_Sun + 1.914666471 * (math.sin(math.radians(M_Sun))) + 0.918994643 * math.sin(
        math.radians(2 * M_Sun))

    """s_star = 1.000140612-(0.016708617*math.cos(math.radians(M_Sun)))-(0.000139589*math.cos(math.radians(2*M_Sun)))"""

    epsilon = 23.439291 - (0.0130042 * T_UT1)

    s_i = [0, math.cos(math.radians(lambda_elliptic)), math.cos(epsilon) * math.sin(math.radians(lambda_elliptic)),
           math.sin(epsilon) * math.sin(math.radians(lambda_elliptic))]


    def quaternion_multiply(quaternion1, quaternion0):
        w0, x0, y0, z0 = quaternion0
        w1, x1, y1, z1 = quaternion1
        return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                         x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                         -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                         x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)


    temp = quaternion_multiply(s_i, q_star)
    s_b = quaternion_multiply(q, temp)

    mod_sb = math.sqrt((s_b[1] ** 2) + (s_b[2] ** 2) + (s_b[3] ** 2))

    print("Sun Vector in ground frame:")
    print(s_b[1:4])

    plot_x.append(delta_t)
    plot_y.append((s_b[3] / mod_sb))

    for i in range(0, 4):
        for j in range(0, 4):
            rate_of_change[i] = 0.5*X[i][j] * Y[j]

plt.plot(plot_x, plot_y)
plt.xlabel("Time\nIn Milliseconds")
plt.ylabel("Direction Cosine withe Z axis")
plt.show()