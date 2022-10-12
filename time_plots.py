import numpy as np
import time
import h5py
import matplotlib.pyplot as plt
import dedalus.public as d3
import matplotlib


# Parameters
nx = 512 # fourier resolution
nz = 64  # chebyshev resolution

H = 10  # depth of water in meters
h_e = 1 #ekman thickness
dh = h_e/H  # dh = ekman thickness divided by H
da = 0.01  # aspect ratio = ratio of height to length
f = 1e-4  # coriolis param in 1/s

L_func = lambda H, delta_a: H / delta_a
L = L_func(H, da)
k = (2 * np.pi) / (L)

Re = 1.6
Rg=0.1

sim_time = 50
dt = 0.125
psi_arr = np.zeros((12,nx, nz))

zcoord = d3.Coordinate('z')
dist = d3.Distributor(zcoord)
z_basis = d3.Chebyshev(zcoord, size=nz, bounds=(0, H))

folder = "snapshots"
with h5py.File(folder + '/' + folder + '_s1/'+ folder + '_s1_p0.h5',
               mode='r') as file:  # reading file

    psi_arr[:, :, :] = np.array(file['tasks']['<psi>'])  # psi

print(psi_arr[8,:,:])

x = np.linspace(0, L, nx)
X, Z = np.meshgrid(dist.local_grid(z_basis),x )

fig,ax= plt.subplots(constrained_layout=True)
CM= plt.pcolormesh(Z, X, psi_arr[11,:,:], shading='gouraud',cmap='PRGn')
cbar = fig.colorbar(CM)
plt.ylabel('vertical depth')
plt.xlabel('periodic x-axis (0,2$\pi$)')
plt.title('$\psi(t=0)$')
plt.show()
plt.close(fig)