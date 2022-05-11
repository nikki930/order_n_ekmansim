import os
import numpy as np
import time
import h5py
import math
import matplotlib.pyplot as plt
import sys
from scipy.fft import fft, fftfreq
from contextlib import suppress

N=100
nx = 512  # fourier resolution
nz = 128  # chebyshev resolution

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
psi_x_arr_corrected = np.zeros((N + 1, nx, nz))
psi_z_arr = np.zeros((N + 1, nx, nz))
psi_z_arr_corrected = np.zeros((N + 1, nx, nz))
zeta_arr = np.zeros((N + 1, nx, nz))
zeta_arr_corrected = np.zeros((N + 1, nx, nz))
zeta_x_arr = np.zeros((N + 1, nx, nz))
zeta_z_arr = np.zeros((N + 1, nx, nz))
v_z_arr = np.zeros((N + 1, nx, nz))
v_x_arr = np.zeros((N + 1, nx, nz))
v_arr = np.zeros((N + 1, nx, nz))
v_arr_corrected = np.zeros((N + 1, nx, nz))
v_x_arr_corrected= np.zeros((N + 1, nx, nz))
v_z_arr_corrected= np.zeros((N + 1, nx, nz))
max_vals = np.zeros(N + 1)



for i in range(N+1):
    folder_n = 'out_' + str(i) + '_n'

    with h5py.File(folder_n + '/' + folder_n + '_s1/' + folder_n + '_s1_p0.h5',
                   mode='r') as file:  # reading file

        psi_arr[i, :, :] = np.array(file['tasks']['<psi>'])  # psi
        zeta_arr[i, :, :] = np.array(file['tasks']['<zeta>'])  # zeta
        psi_x_arr[i, :, :] = np.array(file['tasks']['<psix>'])  # d/dx (psi)
        psi_z_arr[i, :, :] = np.array(file['tasks']['<psiz>'])  # d/dz (psi)
        # zeta_arr[i, :, :] = np.array(file['tasks']['<zeta>'])  # zeta
        # zeta_x_arr[i, :, :] = np.array(file['tasks']['<zetax>'])  # d/dx (zeta)
        # zeta_z_arr[i, :, :] = np.array(file['tasks']['<zetaz>'])  # d/dz (zeta)
        v_x_arr[i, :, :] = np.array(file['tasks']['<vx>'])  #
        v_z_arr[i, :, :] = np.array(file['tasks']['<vz>'])  #
        v_arr[i, :, :] = np.array(file['tasks']['<v>'])  #
    if i == 0:
        psi_arr_corrected[i, :, :] = psi_arr[0, :, :]
        zeta_arr_corrected[i, :, :] = zeta_arr[0, :, :]
        psi_x_arr_corrected[i, :, :] = psi_x_arr[0, :, :]
        psi_z_arr_corrected[i, :, :] = psi_z_arr[0, :, :]
        v_arr_corrected[i, :, :] = v_arr[0, :, :]
        v_x_arr_corrected[i, :, :] = v_x_arr[0, :, :]
        v_z_arr_corrected[i, :, :] = v_z_arr[0, :, :]
    else:
        psi_arr_corrected[i, :, :] = psi_arr_corrected[i - 1, :, :] + psi_arr[i, :, :]
        zeta_arr_corrected[i, :, :] = zeta_arr_corrected[i - 1, :, :] + zeta_arr[i, :, :]
        psi_x_arr_corrected[i, :, :] = psi_x_arr_corrected[i - 1, :, :] + psi_x_arr[i, :, :]
        psi_z_arr_corrected[i, :, :] = psi_z_arr_corrected[i - 1, :, :] + psi_z_arr[i, :, :]
        v_arr_corrected[i, :, :] = v_arr_corrected[i - 1, :, :] + v_arr[i, :, :]
        v_x_arr_corrected[i, :, :] =v_x_arr_corrected[i - 1, :, :] + v_x_arr[i, :, :]
        v_z_arr_corrected[i, :, :] = v_z_arr_corrected[i - 1, :, :] + v_z_arr[i, :, :]

x = np.linspace(0, L, nx)
z = np.linspace(0, H, nz)
X, Z = np.meshgrid(z, x)
fig,ax= plt.subplots(constrained_layout=True)
order = 2
CS = plt.contour(Z, X, psi_arr_corrected[order, :, :], 30, colors='k')
plt.clabel(CS, CS.levels[1::5],inline=1,fontsize=5)
CM= plt.pcolormesh(Z, X, psi_x_arr[order, :, :], shading='gouraud',cmap='PRGn')

#ax.set_title('Gouraud Shading')
cbar = fig.colorbar(CM)
cbar.ax.set_ylabel('Velocity Field')

plt.ylabel('vertical depth')
plt.xlabel('periodic x-axis (0,2$\pi$)')
plt.title('dw -velocity versus psi(m/s)')

#plt.savefig('u_overlaid.png')
plt.show()
plt.close(fig)