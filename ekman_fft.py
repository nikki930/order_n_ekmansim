import os
import numpy as np
from tqdm import tqdm
from tqdm.notebook import tqdm_notebook
import time
import h5py
import math
import matplotlib.pyplot as plt
import sys

nx = 256  # fourier resolution
nz = 512  # chebyshev resolution

H = 10  # depth of water in meters
h_e = 1 #ekman thickness

dh = h_e/H  # dh = ekman thickness divided by H
da = 0.01  # aspect ratio = ratio of height to length
f = 1e-4  # coriolis param in 1/s

N = 50 # the order up to which the model runs


L_func = lambda H, delta_a: H / delta_a
L = L_func(H, da)

x = np.linspace(0, L, nx)
z = np.linspace(0, H, nz) #there are nz elements in z, we cant the slice at about 0.7*nz element

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

print(z[358])
for i in range (N+1):
    folder_n = 'out_' + str(i) + '_n'

    with h5py.File(folder_n + '/' + folder_n + '_s1/' + folder_n + '_s1_p0.h5',
                   mode='r') as file:  # reading file

        psi_arr[i, :, :] = np.array(file['tasks']['<psi>'])  # psi
        zeta_arr[i, :, :] = np.array(file['tasks']['<zeta>'])  # zeta
        v_arr[i, :, :] = np.array(file['tasks']['<v>'])  #



