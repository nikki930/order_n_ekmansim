import os
import numpy as np
import time
import h5py
import math
import matplotlib.pyplot as plt
import sys
from scipy.fft import fft, fftfreq
from contextlib import suppress

N=50
nx = 256  # fourier resolution
nz = 512  # chebyshev resolution

H = 10  # depth of water in meters
h_e = 1 #ekman thickness
dh = h_e/H  # dh = ekman thickness divided by H
da = 0.01  # aspect ratio = ratio of height to length
f = 1e-4  # coriolis param in 1/s
L_func = lambda H, delta_a: H / delta_a
L = L_func(H, da)

psi_arr = np.zeros((N + 1, nx, nz))
psi_arr_corrected = np.zeros((N + 1, nx, nz))
psi_x_arr = np.zeros((N + 1, nx, nz))
psi_z_arr = np.zeros((N + 1, nx, nz))
zeta_arr = np.zeros((N + 1, nx, nz))
zeta_x_arr = np.zeros((N + 1, nx, nz))
zeta_z_arr = np.zeros((N + 1, nx, nz))
v_z_arr = np.zeros((N + 1, nx, nz))
v_x_arr = np.zeros((N + 1, nx, nz))
v_arr = np.zeros((N + 1, nx, nz))
max_vals = np.zeros(N + 1)

for i in range(N+1):
    folder_n = 'out_' + str(i) + '_n'

    with h5py.File(folder_n + '/' + folder_n + '_s1/' + folder_n + '_s1_p0.h5',
                   mode='r') as file:  # reading file

        psi_arr[i, :, :] = np.array(file['tasks']['<psi>'])  # psi
        zeta_arr[i, :, :] = np.array(file['tasks']['<zeta>'])  # zeta
        psi_x_arr[i, :, :] = np.array(file['tasks']['<psix>'])  # d/dx (psi)
        psi_z_arr[i, :, :] = np.array(file['tasks']['<psiz>'])  # d/dz (psi)
        zeta_arr[i, :, :] = np.array(file['tasks']['<zeta>'])  # zeta
        zeta_x_arr[i, :, :] = np.array(file['tasks']['<zetax>'])  # d/dx (zeta)
        zeta_z_arr[i, :, :] = np.array(file['tasks']['<zetaz>'])  # d/dz (zeta)
        v_x_arr[i, :, :] = np.array(file['tasks']['<vx>'])  #
        v_z_arr[i, :, :] = np.array(file['tasks']['<vz>'])  #
        v_arr[i, :, :] = np.array(file['tasks']['<v>'])  #

x = np.linspace(0, L, nx)
z = np.linspace(0, H, nz)
psi_slice = lambda i:psi_arr[i,:,38]
print(z[38])

xf = fftfreq(int(nx/2), 1 / 20)
psi_f = fft(psi_slice(10))
power_psi = lambda i: (fft(psi_slice(i)[:int(nx/2)]))
plt.plot(xf, power_psi(0), label = 'order 0')
plt.plot(xf, power_psi(1), label = 'order 10')
plt.plot(xf, power_psi(3), label = 'order 30')
plt.xlabel('')
plt.title("Power Spectra")
plt.legend()
plt.show()