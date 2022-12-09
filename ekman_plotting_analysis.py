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
nz = 68  # chebyshev resolution

H = 200  # depth of water in meters
h_e = 20 #ekman thickness
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
max_vals = np.zeros(N + 1)



for i in range(N+1):
    folder_n = 'out_' + str(i) + '_n'

    with h5py.File('Re_big/' + folder_n + '/' + folder_n + '_s1/' + folder_n + '_s1_p0.h5',
                   mode='r') as file:  # reading file

        psi_arr[i, :, :] = np.array(file['tasks']['<psi>'])  # psi
        zeta_arr[i, :, :] = np.array(file['tasks']['<zeta>'])  # zeta
        psi_x_arr[i, :, :] = np.array(file['tasks']['<psix>'])  # d/dx (psi)
        psi_z_arr[i, :, :] = np.array(file['tasks']['<psiz>'])  # d/dz (psi)
        # zeta_arr[i, :, :] = np.array(file['tasks']['<zeta>'])  # zeta
        # zeta_x_arr[i, :, :] = np.array(file['tasks']['<zetax>'])  # d/dx (zeta)
        # zeta_z_arr[i, :, :] = np.array(file['tasks']['<zetaz>'])  # d/dz (zeta)
        # v_x_arr[i, :, :] = np.array(file['tasks']['<vx>'])  #
        # v_z_arr[i, :, :] = np.array(file['tasks']['<vz>'])  #
        v_arr[i, :, :] = np.array(file['tasks']['<v>'])  #
    if i == 0:
        psi_arr_corrected[i, :, :] = psi_arr[0, :, :]
        zeta_arr_corrected[i, :, :] = zeta_arr[0, :, :]
        psi_x_arr_corrected[i, :, :] = psi_x_arr[0, :, :]
        psi_z_arr_corrected[i, :, :] = psi_z_arr[0, :, :]
        v_arr_corrected[i, :, :] = v_arr[0, :, :]
    else:
        psi_arr_corrected[i, :, :] = psi_arr_corrected[i - 1, :, :] + psi_arr[i, :, :]
        zeta_arr_corrected[i, :, :] = zeta_arr_corrected[i - 1, :, :] + zeta_arr[i, :, :]
        psi_x_arr_corrected[i, :, :] = psi_x_arr_corrected[i - 1, :, :] + psi_x_arr[i, :, :]
        psi_z_arr_corrected[i, :, :] = psi_z_arr_corrected[i - 1, :, :] + psi_z_arr[i, :, :]
        v_arr_corrected[i, :, :] = v_arr_corrected[i - 1, :, :] + v_arr[i, :, :]
x = np.linspace(0, L, nx)
z = np.linspace(0, H, nz)

psi_slice = lambda i: psi_arr_corrected[i,:,38]
zeta_slice = lambda i: zeta_arr_corrected[i,:,38]
w_slice = lambda i: - psi_x_arr_corrected[i,:,38]
u_slice = lambda i: psi_z_arr_corrected[i,:,38]
v_slice = lambda i: v_arr_corrected[i,:,38]

def DFT(x):
    """
    Function to calculate the
    discrete Fourier Transform
    of a 1D real-valued signal x
    """
    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))
    e = np.exp(-2j * np.pi * k * n / N)

    X = np.dot(e, x)

    return X

xf = fftfreq(int(nx/2), 1 / int(nx/2))

def fft_out(data):
    out = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(data)))
    conj_out = np.conj(np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(data))))

    return (out * conj_out)

KE_total = lambda k: (fft_out(u_slice(k))[int(nx / 2 + 1):] + fft_out(w_slice(k))[int(nx / 2 + 1):] + fft_out(v_slice(k))[int(nx / 2 + 1):]) / 2
#power_psi = lambda i: np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(psi_slice(i))))
#conj_power_psi = lambda i: np.conj(np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(psi_slice(i)))))


orders_plt=np.arange(0,N)

#orders_plt=20
#plt.loglog(np.arange(1, int(nx/2)), fft_out(psi_slice(orders_plt))[int(nx/2 + 1):], label='order' + str(orders_plt))

for k in orders_plt:
    #plt.loglog(np.arange(1, int(nx/2)), fft_out(psi_slice(k))[int(nx/2 + 1):], label='order' + str(k))
    #plt.loglog(np.arange(1,int(nx/2)), fft_out(zeta_slice(k))[int(nx/2 + 1):], label='order' + str(k))
    #plt.loglog(np.arange(1, int(nx / 2)), (fft_out(w_slice(k))[int(nx / 2 + 1):])/2, label='order' + str(k))
    #plt.loglog(np.arange(1, int(nx / 2)), (fft_out(u_slice(k))[int(nx / 2 + 1):]) / 2, label='order' + str(k))
    plt.plot(np.arange(0, 512), u_slice(k), label='order' + str(k))
    #plt.plot(np.arange(1, int(nx / 2)), KE_total(k), label='order' + str(k))

    #plt.plot(np.arange(0,int(nx/2)),(DFT(psi_slice(k)) * np.conj(DFT(psi_slice(k)))))

#plt.yscale('log')
plt.ylabel('streamfunction strength (kg/ms)')
plt.xlabel('x-grid points')
plt.grid(color='green', linestyle='--', linewidth=0.5)
#plt.title("Total Kinetic Energy Spectrum")
plt.title("$\psi_{max}$ Horizontal Slice for orders n $\in$ [0,100]")
#plt.legend()
plt.savefig('KE_all')
plt.show()