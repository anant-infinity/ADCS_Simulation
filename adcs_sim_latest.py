# -*- coding: utf-8 -*-
"""
Created on Tue May 29 19:22:38 2018

@author: Anant
"""

import random
import math
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from astropy.time import Time
from datetime import  timedelta
import poliastro.twobody
import poliastro
from poliastro.plotting import plot
#plt.style.use("seaborn")

# initial date
year = 2019
month = 9
day = 1
hour = 0
minute = 0
second = 0

plot_y = []
plot_x = []

plot_t = []

plot_Tg1 = []
plot_Tg2 = []
plot_Tg3 = []

plot_omega1 = []
plot_omega2 = []
plot_omega3 = []

plot_alpha1 = []
plot_alpha2 = []
plot_alpha3 = []

# Random Quaternion
q = [0, 0, 0, 0]
for i in range(0, 4):
    q[i] = random.uniform(0, 1)  # initial quaternion values between 0 and 1


# 4x4 matrix
X = [[q[0], -q[1], -q[2], -q[3]],
     [q[1], q[0], -q[3], q[1]],
     [q[2], q[3], q[0], -q[1]],
     [q[3], -q[2], q[1], q[0]]]
# 4x1 matrix - Initial omega
omega = [0, 0.34906585, 0.34906585, 0.349065850]
omega_three = omega[1:4]

# 4x1 matrix - rate_of_change of quaternion
rate_of_change = [0, 0, 0, 0]

for i in range(0, 4):
    for j in range(0, 4):
        rate_of_change[i] += X[i][j] * omega[j]

print("Initial rate of change is: ", rate_of_change)
print()

# defining orbit
# initial classical orbital elements
# noinspection PyUnresolvedReferences
a = 6878.137 * u.km
ecc = 0 * u.one
#inc = 0.872699533 * u.rad
#raan = 6.280724393 * u.rad
#argp = 0 * u.rad
#nu = 6.280986192 * u.rad
# noinspection PyUnresolvedReferences
inc = 50.002 * u.deg
# noinspection PyUnresolvedReferences
raan = 359.859 * u.deg
# noinspection PyUnresolvedReferences
argp = 0 * u.deg
# noinspection PyUnresolvedReferences
nu = 359.874 * u.deg
date_epoch = Time("2019-09-01 00:00", scale='utc')
ss = Orbit.from_classical(Earth, a, ecc, inc, raan, argp, nu, date_epoch)
# 2448122.5

# Simulation Loop Starts
for delta_t in range(0, 60000, 100):
    print("Quaternion after ", delta_t, "milliseconds is: "),
    for i in range(0, 4):
        q[i] = q[i] + (rate_of_change[i] * delta_t * 0.001)
    mod_q = math.sqrt(q[0] ** 2 + q[1] ** 2 + q[2] ** 2 + q[3] ** 2)
    for i in range(0, 4):
        q[i] = q[i] / mod_q
        print(" ", q[i], " "),
    print("")
    print()
    # q conjugate
    q_star = [q[0], -q[1], -q[2], -q[3]]

    X = [[q[0], -q[1], -q[2], -q[3]],
         [q[1], q[0], -q[3], q[1]],
         [q[2], q[3], q[0], -q[1]],
         [q[3], -q[2], q[1], q[0]]]

    # Julian Date from current Date
    JD = 367 * year - int(7 * (year + int((month + 9) / 12))) + int(275 * month / 9) + day + 1721013.5 + (hour / 24) + (
            minute / 14400) + ((second + (delta_t*0.001)) / 86400)

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

    print("Sun Vector in body frame is: ", s_b[1:4])
    print()

    #Adding values to plot of Direction Cosine
    plot_x.append(delta_t)
    plot_y.append((s_b[3] / mod_sb))

    def get_Rc(ss):
        #
        #poliastro.twobody.propagation.mean_motion(ss, delta_t/1000)

        #t = timedelta(seconds=(delta_t/1000))
        # noinspection PyUnresolvedReferences
        #t = timedelta(seconds = delta_t)
        #print (t)

        t_inmins = (delta_t/60000)
        # noinspection PyUnresolvedReferences
        ss = ss.propagate(t_inmins * u.min)
        Rc = ss.state.r.value
        print(ss.epoch)

        # noinspection PyUnresolvedReferences
        #ss_new = ss.propagate(delta_t/1000 * u.s)
        #print("Orbit Details: ", ss_new, ss_new.epoch)

        # Convert to Meters
        for i in range(3):
            Rc[i]=Rc[i]*1000
        return Rc


    def gravity_gradient_torque(I , Rc):

        Tg = [0, 0, 0]

        # Constant 3*G*Me
        k = 3 * 6.674 * (10 ** (-11)) * 5.972 * (10 ** 24)

        #mod_Rc = math.sqrt(Rc[0] ** 2 + Rc[1] ** 2 + Rc[2] ** 2)
        ##for var in range(3):
            #Rc_norm[var] = (Rc[var]/mod_Rc)
        mod_Rc = math.sqrt((Rc[0] ** 2) + (Rc[1] ** 2) + (Rc[2] ** 2))

        L1 = [0, Rc[0], Rc[1], Rc[2]]
        temp4 = quaternion_multiply(L1, q_star)
        Rc_b = quaternion_multiply(q, temp4)

        print("Rc in Body Frame is:", Rc_b)
        print("Mod of Rc is: ", mod_Rc)
        print()


        # Randomly generated values of Rc1, Rc2 , Rc3
        #Rc1 = random.uniform(0, 1)
        #Rc2 = random.uniform(0, 1)
        #Rc3 = random.uniform(0, 1)

        Tg[0] = ((k * Rc_b[2] * Rc_b[3] * (I[2][2] - I[1][1])) / (mod_Rc ** 5))
        Tg[1] = ((k * Rc_b[1] * Rc_b[3] * (I[0][0] - I[2][2])) / (mod_Rc ** 5))
        Tg[2] = ((k * Rc_b[1] * Rc_b[2] * (I[1][1] - I[0][0])) / (mod_Rc ** 5))

        return Tg

    # Euler's Formula
    def get_alpha(I, Tg, omega):

        temp1 = [0, 0, 0]

        for i in range(3):
            for k in range(3):
                temp1[i] += I[i][k] * omega[k]

        # cross product
        temp2 = [omega[1] * temp1[2] - omega[2] * temp1[1],
                 omega[2] * temp1[0] - omega[0] * temp1[2],
                 omega[0] * temp1[1] - omega[1] * temp1[0]]

        temp3 = [0, 0, 0]
        for i in range(0, 3):
            temp3[i] = Tg[i] - temp2[i]

        I_num = np.matrix(I)

        #print (I_num)
        #print ("Inverse is: ",I_num.I)

        I_inverse = I_num.I.tolist()

        alpha = [0, 0, 0]

        for i in range(3):
            for k in range(3):
                alpha[i] += I_inverse[i][k] * temp3[k]

        return alpha


    # MI got from Spencer
    I = [[0.09, 0, 0], [0, 0.12, 0], [0, 0, 0.14]]

    # Moment of Inertia - Assuming uniform cuboid and assuming principal axes (approx.)
    # I = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

    # I[0][0] = 10 * ((0.2 ** 2) + (0.3 ** 2)) / 12
    # I[1][1] = 10 * ((0.45 ** 2) + (0.3 ** 2)) / 12
    # I[2][2] = 10 * ((0.2 ** 2) + (0.45 ** 2)) / 12

    L2 = get_Rc(ss)
    print("Rc in ECI Frame in meters: ", L2)
    print()

    Tg = gravity_gradient_torque(I, L2)
    print("Torque is :", Tg)
    print()

    plot_Tg1.append(Tg[0])
    plot_Tg2.append(Tg[1])
    plot_Tg3.append(Tg[2])

    alpha = get_alpha(I, Tg, omega_three)
    print("Current alpha is: ", alpha)

    plot_alpha1.append(alpha[0] * 57.2958)
    plot_alpha2.append(alpha[1] * 57.2958)
    plot_alpha3.append(alpha[2] * 57.2958)

    for i in range(0,3):
        omega[i+1] = omega[i+1] + (alpha[i]*0.1)
        omega_three[i] = omega_three[i] + (alpha[i]*0.1)

    print("Current Omega is", omega_three)
    print()

    plot_omega1.append(omega_three[0] * 57.2958)
    plot_omega2.append(omega_three[1] * 57.2958)
    plot_omega3.append(omega_three[2] * 57.2958)

    plot_t.append(delta_t)

    #updating rate of change of quaternion
    for i in range(0, 4):
        rate_of_change[i] = 0
        for j in range(0, 4):
            rate_of_change[i] += 0.5*X[i][j] * omega[j]

print (plot_t)
print(plot_Tg1)
plt.subplot(3, 1, 1)
plt.plot(plot_t, plot_Tg1, 'b-' , label='Torque about x')
plt.plot(plot_t, plot_Tg2, 'r-' , label='Torque about y')
plt.plot(plot_t, plot_Tg3, 'g-' , label='Torque about z')
plt.title('Tumbling  Simulation')
plt.ylabel('Torque')
plt.legend(loc='best')

plt.subplot(3, 1, 2)
plt.plot(plot_t, plot_alpha1, 'b-' , label='Alpha about x')
plt.plot(plot_t, plot_alpha2, 'r-' , label='Alpha about y')
plt.plot(plot_t, plot_alpha3, 'g-' , label='Alpha about z')
plt.ylabel('Angular Acceleration\ndegree/sec^2')
plt.legend(loc='best')

plt.subplot(3, 1, 3)
plt.plot(plot_t, plot_omega1, 'b-', label='omega about x')
plt.plot(plot_t, plot_omega2, 'r-', label='omega about y')
plt.plot(plot_t, plot_omega3, 'g-', label='omega about z')
plt.ylabel('angular velocity\ndegree/s')
plt.legend(loc='best')
plt.xlabel("Time\nIn Milliseconds")


plt.show()

# plt.plot(plot_x, plot_y)
# plt.xlabel("Time\nIn Milliseconds")
# plt.ylabel("Direction Cosine withe Z axis")
# plt.show()