
"""

This is the file where it takes the solver run folder, loads all outputed variabled that the solver file computed, and
from which various analysis plots are made:

- fourier power spectra
- forcing analysis for zeta eq'n
- forcing analysis for v eq'n

"""

import os
import numpy as np
import time
import h5py
import math
import matplotlib.pyplot as plt
import sys
from scipy.fft import fft, fftfreq
from dedalus import public as d3
from contextlib import suppress

########################## MUST MATCH MASTER RUN FILE ##################################################################
N=100
nx = 512  # fourier resolution
nz = 128  # chebyshev resolution
run_folder =  'Re_big/'

H = 10  # depth of water in meters
h_e = 1 #ekman thickness
dh = h_e/H  # dh = ekman thickness divided by H
da = 0.01  # aspect ratio = ratio of height to length
f = 1e-4  # coriolis param in 1/s
L_func = lambda H, delta_a: H / delta_a
L = L_func(H, da)

J_v_arr = np.zeros((N + 1, nx, nz))
J_zeta_arr = np.zeros((N + 1, nx, nz))
J_v_arr_corrected = np.zeros((N + 1, nx, nz))
J_zeta_arr_corrected = np.zeros((N + 1, nx, nz))
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
v_z_arr = np.zeros((N + 1, nx, nz))
v_z_arr_corrected = np.zeros((N + 1, nx, nz))
v_arr = np.zeros((N + 1, nx, nz))
v_arr_corrected = np.zeros((N + 1, nx, nz))
max_vals = np.zeros(N + 1)


# folder_n = run_folder + 'out_' + str(i) + '_n'
# folder_n_sub = 'out_' + str(self.n) + '_n'
########################################################################################################################
for i in range(N+1):
    folder_n = run_folder +'out_' + str(i) + '_n'
    folder_n_sub = 'out_' + str(i) + '_n'
    with h5py.File(folder_n + '/' + folder_n_sub + '_s1/' + folder_n_sub + '_s1_p0.h5',
                   mode='r') as file:  # reading file

        psi_arr[i, :, :] = np.array(file['tasks']['<psi>'])  # psi
        zeta_arr[i, :, :] = np.array(file['tasks']['<zeta>'])  # zeta
        psi_x_arr[i, :, :] = np.array(file['tasks']['<psix>'])  # d/dx (psi)
        psi_z_arr[i, :, :] = np.array(file['tasks']['<psiz>'])  # d/dz (psi)
        # zeta_arr[i, :, :] = np.array(file['tasks']['<zeta>'])  # zeta
        # zeta_x_arr[i, :, :] = np.array(file['tasks']['<zetax>'])  # d/dx (zeta)
        # zeta_z_arr[i, :, :] = np.array(file['tasks']['<zetaz>'])  # d/dz (zeta)
        # v_x_arr[i, :, :] = np.array(file['tasks']['<vx>'])  #
        v_z_arr[i, :, :] = np.array(file['tasks']['<vz>'])  #
        v_arr[i, :, :] = np.array(file['tasks']['<v>'])  #
        J_v_arr[i, :, :] = np.array(file['tasks']['<J_psi_v>'])
        J_zeta_arr[i, :, :] = np.array(file['tasks']['<J>'])
    if i == 0:
        psi_arr_corrected[i, :, :] = psi_arr[0, :, :]
        zeta_arr_corrected[i, :, :] = zeta_arr[0, :, :]
        psi_x_arr_corrected[i, :, :] = psi_x_arr[0, :, :]
        psi_z_arr_corrected[i, :, :] = psi_z_arr[0, :, :]
        v_arr_corrected[i, :, :] = v_arr[0, :, :]
        v_z_arr[i, :, :] = v_z_arr[0,:,:]
        J_v_arr_corrected[i, :, :] = J_v_arr[0, :, :]
        J_zeta_arr_corrected[i, :, :] = J_zeta_arr[0, :, :]
    else:
        psi_arr_corrected[i, :, :] = psi_arr_corrected[i - 1, :, :] + psi_arr[i, :, :]
        zeta_arr_corrected[i, :, :] = zeta_arr_corrected[i - 1, :, :] + zeta_arr[i, :, :]
        psi_x_arr_corrected[i, :, :] = psi_x_arr_corrected[i - 1, :, :] + psi_x_arr[i, :, :]
        psi_z_arr_corrected[i, :, :] = psi_z_arr_corrected[i - 1, :, :] + psi_z_arr[i, :, :]
        v_arr_corrected[i, :, :] = v_arr_corrected[i - 1, :, :] + v_arr[i, :, :]
        v_z_arr_corrected[i, :, :] = v_z_arr_corrected[i - 1, :, :] + v_z_arr[i, :, :]
        J_v_arr_corrected[i, :, :] = J_v_arr_corrected[i-1, :, :] + J_v_arr[i, :, :]
        J_zeta_arr_corrected[i, :, :] = J_zeta_arr_corrected[i-1, :, :] + J_zeta_arr[i, :, :]

x = np.linspace(0, L, nx)
z = np.linspace(0, H, nz)

#______________________________________FOURIER ANALYSIS: ______________________________________________________________
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

#______________________________________ZETA FORCING TEST:____________________________________________________

psi_A_arr_test=np.zeros((nx,nz))
zeta_A_arr_test=np.zeros((nx,nz))
psi_B_arr_test=np.zeros((nx,nz))
zeta_B_arr_test=np.zeros((nx,nz))

folder = 'Forcing_Analysis/'
folder_n = 'Forcing_Analysis/' + 'out'
folder_n_sub = 'out'

#plotting
with h5py.File(folder_n + '/' + folder_n_sub + '_s1/' + folder_n_sub + '_s1_p0.h5',
               mode='r') as file:  # reading file

    psi_A_arr_test[:, :] = np.array(file['tasks']['<psi_A>']) # psi
    zeta_A_arr_test[:, :] = np.array(file['tasks']['<zeta_A>'])  # zeta
    psi_B_arr_test[:, :] = np.array(file['tasks']['<psi_B>'])  # psi
    zeta_B_arr_test[:, :] = np.array(file['tasks']['<zeta_B>'])  # zeta


zcoord = d3.Coordinate('z')
dist = d3.Distributor(zcoord)
z_basis = d3.Chebyshev(zcoord, size=nz, bounds=(0, H))
x = np.linspace(0, L, nx)
X, Z = np.meshgrid(dist.local_grid(z_basis),x )

fig,ax= plt.subplots(constrained_layout=True)
CS = plt.contour(Z, X, psi_A_arr_test, 30,cmap='PRGn',vmin=-3e-3,vmax=3e-3)
plt.clabel(CS,inline=1,fontsize=5)
#CM= plt.pcolormesh(Z, X, psi_A_arr_test, shading='gouraud',cmap='PRGn',vmin=-3e-3,vmax=3e-3)
#CM= plt.pcolormesh(Z, X, psi_A_arr_test, shading='gouraud',cmap='PRGn',vmin=-9.6e-3,vmax=9.6e-3)
cbar = fig.colorbar(CS)
cbar.ax.set_ylabel('Streamfunction (kg/ms)')
plt.ylabel('vertical depth')
plt.xlabel('periodic x-axis (0,2$\pi$)')
plt.title('$\psi_A$ (forced by $-fv_z$)')
plt.savefig(folder + 'psi_A.png')
plt.close(fig)

fig,ax= plt.subplots(constrained_layout=True)
CS = plt.contour(Z, X, psi_B_arr_test, 30,cmap='PRGn',vmin=-3e-3,vmax=3e-3)
plt.clabel(CS,inline=1,fontsize=5)
#CM= plt.pcolormesh(Z, X, psi_B_arr_test, shading='gouraud',cmap='PRGn',vmin=-3e-3,vmax=3e-3)
#CM= plt.pcolormesh(Z, X, psi_B_arr_test, shading='gouraud',cmap='PRGn', vmin=-9.6e-3,vmax=9.6e-3)
cbar = fig.colorbar(CS)
cbar.ax.set_ylabel('Streamfunction (kg/ms)')
plt.ylabel('vertical depth')
plt.xlabel('periodic x-axis (0,2$\pi$)')
plt.title('$\psi_B$ (forced by $J(\psi,\zeta)$)')
plt.savefig(folder + 'psi_B.png')
plt.close(fig)

fig,ax= plt.subplots(constrained_layout=True)
CS = plt.contour(Z, X, psi_A_arr_test, 30, colors='k',vmin=-1.1e-3,vmax=1.1e-3)
plt.clabel(CS, CS.levels[1::5],inline=1,fontsize=5)
CM= plt.pcolormesh(Z, X, zeta_A_arr_test, shading='gouraud',cmap='PRGn', vmin=-1.1e-3,vmax=1.1e-3)
#CM= plt.pcolormesh(Z, X, zeta_A_arr_test, shading='gouraud',cmap='PRGn', vmin=-2.4e-3,vmax=2.4e-3)
cbar = fig.colorbar(CM)
cbar.ax.set_ylabel('Vorticity ($s^{-1}$)')
plt.ylabel('vertical depth')
plt.xlabel('periodic x-axis (0,2$\pi$)')
plt.title('$\zeta_A$ (forced by $-fv_z$)')
plt.savefig(folder + 'zeta_A.png')
plt.close(fig)

fig,ax= plt.subplots(constrained_layout=True)
CS = plt.contour(Z, X, psi_B_arr_test, 30, colors='k',vmin=-1.1e-3,vmax=1.1e-3)
plt.clabel(CS, CS.levels[1::5],inline=1,fontsize=5)
#CM= plt.pcolormesh(Z, X, zeta_B_arr_test, shading='gouraud',cmap='PRGn', vmin=-2.4e-3,vmax=2.4e-3)
CM= plt.pcolormesh(Z, X, zeta_B_arr_test, shading='gouraud',cmap='PRGn', vmin=-1.1e-3,vmax=1.1e-3)
cbar = fig.colorbar(CM)
cbar.ax.set_ylabel('Vorticity ($s^{-1}$)')
plt.ylabel('vertical depth')
plt.xlabel('periodic x-axis (0,2$\pi$)')
plt.title('$\zeta_B$ (forced by $J(\psi,\zeta)$)')
plt.savefig(folder + 'zeta_B.png')
plt.close(fig)

#______________________________________V FORCING TEST:____________________________________________________

v_A_arr_test=np.zeros((nx,nz))
v_Az_arr_test=np.zeros((nx,nz))
v_B_arr_test=np.zeros((nx,nz))
v_Bz_arr_test=np.zeros((nx,nz))

folder = 'Forcing_Analysis/'
folder_n = folder+'v_out'
folder_n_sub = 'v_out'

#plotting
with h5py.File(folder_n + '/' + folder_n_sub + '_s1/' + folder_n_sub + '_s1_p0.h5',
               mode='r') as file:  # reading file

    v_A_arr_test[:, :] = np.array(file['tasks']['<v_A>'])
    v_Az_arr_test[:, :] = np.array(file['tasks']['<v_Az>'])
    v_B_arr_test[:, :] = np.array(file['tasks']['<v_B>'])
    v_Bz_arr_test[:, :] = np.array(file['tasks']['<v_Bz>'])


zcoord = d3.Coordinate('z')
dist = d3.Distributor(zcoord)
z_basis = d3.Chebyshev(zcoord, size=nz, bounds=(0, H))
x = np.linspace(0, L, nx)
X, Z = np.meshgrid(dist.local_grid(z_basis),x )

fig,ax= plt.subplots(constrained_layout=True)
CS = plt.contour(Z, X, zeta_A_arr_test, 30, colors='k')
plt.clabel(CS,inline=1,fontsize=5)
CM= plt.pcolormesh(Z, X, v_A_arr_test, shading='gouraud',cmap='PRGn',vmin=-8e-3,vmax=8e-3)
#CM= plt.pcolormesh(Z, X, psi_A_arr_test, shading='gouraud',cmap='PRGn',vmin=-9.6e-3,vmax=9.6e-3)
cbar = fig.colorbar(CM)
cbar.ax.set_ylabel('Streamfunction (kg/ms)')
plt.ylabel('vertical depth')
plt.xlabel('periodic x-axis (0,2$\pi$)')
plt.title('$v_A$ (forced by $fu$) with $\zeta_A$ contours')
plt.savefig(folder + 'v_A.png')
plt.close(fig)

fig,ax= plt.subplots(constrained_layout=True)
CS = plt.contour(Z, X, zeta_B_arr_test,30, colors='k')
plt.clabel(CS,inline=1,fontsize=5)
CM= plt.pcolormesh(Z, X, v_B_arr_test, shading='gouraud',cmap='PRGn',vmin=-8e-3,vmax=8e-3)
#CM= plt.pcolormesh(Z, X, psi_B_arr_test, shading='gouraud',cmap='PRGn', vmin=-9.6e-3,vmax=9.6e-3)
cbar = fig.colorbar(CM)
cbar.ax.set_ylabel('Streamfunction (kg/ms)')
plt.ylabel('vertical depth')
plt.xlabel('periodic x-axis (0,2$\pi$)')
plt.title('$v_B$ (forced by $J(psi,v)$) with $\zeta_B$ contours')
plt.savefig(folder + 'v_B.png')
plt.close(fig)

fig,ax= plt.subplots(constrained_layout=True)
CS = plt.contour(Z, X, zeta_A_arr_test, 30, colors='k')
plt.clabel(CS, CS.levels[1::5],inline=1,fontsize=5)
CM= plt.pcolormesh(Z, X, v_Az_arr_test, shading='gouraud',cmap='PRGn')
#CM= plt.pcolormesh(Z, X, zeta_A_arr_test, shading='gouraud',cmap='PRGn', vmin=-2.4e-3,vmax=2.4e-3)
cbar = fig.colorbar(CM)
cbar.ax.set_ylabel('Vorticity ($s^{-1}$)')
plt.ylabel('vertical depth')
plt.xlabel('periodic x-axis (0,2$\pi$)')
plt.title('$vz_A$ (forced by $fu$) with $\zeta_A$ contours')
plt.savefig(folder + 'v_Az.png')
plt.close(fig)

fig,ax= plt.subplots(constrained_layout=True)
CS = plt.contour(Z, X, zeta_B_arr_test, 30, colors='k')
plt.clabel(CS, CS.levels[1::5],inline=1,fontsize=5)
#CM= plt.pcolormesh(Z, X, zeta_B_arr_test, shading='gouraud',cmap='PRGn', vmin=-2.4e-3,vmax=2.4e-3)
CM= plt.pcolormesh(Z, X, v_Bz_arr_test, shading='gouraud',cmap='PRGn')
cbar = fig.colorbar(CM)
cbar.ax.set_ylabel('Vorticity ($s^{-1}$)')
plt.ylabel('vertical depth')
plt.xlabel('periodic x-axis (0,2$\pi$)')
plt.title('$vz_B$ (forced by $J(\psi,v)$) with $\zeta_B$ contours')
plt.savefig(folder + 'v_Bz.png')
plt.close(fig)



"""

############################################################
#FOURIER ANALYSIS:
############################################################

orders_plt=np.arange(0,N,1)
#orders_plt=20
#plt.loglog(np.arange(1, int(nx/2)), fft_out(psi_slice(orders_plt))[int(nx/2 + 1):], label='order' + str(orders_plt))

for k in orders_plt:
    #plt.loglog(np.arange(1, int(nx/2)), fft_out(psi_slice(k))[int(nx/2 + 1):], label='order' + str(k))
    #plt.loglog(np.arange(1,int(nx/2)), fft_out(zeta_slice(k))[int(nx/2 + 1):], label='order' + str(k))
    #plt.loglog(np.arange(1, int(nx / 2)), (fft_out(w_slice(k))[int(nx / 2 + 1):])/2, label='order' + str(k))
    #plt.loglog(np.arange(1, int(nx / 2)), (fft_out(u_slice(k))[int(nx / 2 + 1):]) / 2, label='order' + str(k))
    #plt.plot(np.arange(1, int(nx / 2)), KE_total(k), label='order' + str(k))
    plt.plot(np.arange(0,len(psi_slice(k))), psi_slice(k), label='order' + str(k))

    #plt.plot(np.arange(0,int(nx/2)),(DFT(psi_slice(k)) * np.conj(DFT(psi_slice(k)))))
# plt.yscale("log")
# plt.xlabel('modes')
# plt.title("Total Kinetic Energy Spectrum")
# plt.legend()
# plt.savefig(run_folder + 'KE_psd_notlog')
# plt.show()

plt.ylabel("streamfunction strength (kg/ms)")
plt.xlabel('horizontal grid points')
plt.title("$\psi^n_{max}$ Horizontal Slice for all $n \in [0,100]$")
plt.savefig(run_folder + 'slice_sinusoids')
plt.show()


"""
