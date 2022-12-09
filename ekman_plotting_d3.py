"""

This file is based on the ekman_plotting.py file, with changes made for it to run in the dedalus version 3 environment.
Although dedalus3 is a beta version, it is a version which allows for easy plotting in the basis the problem was solved in,
which rectifies the previous problem of plotting the chebyshev-resoled z axis on a non-chebysehv vertical grid.

This file is just for plotting output variables and terms from the solver at different orders and run folders

"""


import numpy as np
import time
import h5py
import matplotlib.pyplot as plt
import dedalus.public as d3
import matplotlib
#matplotlib.rc('text', usetex=True)
#matplotlib.rc('text.latex', preamble=r'\usepackage{amsmath}')
########################## MUST MATCH MASTER RUN FILE ##################################################################
N=100
#run_folder =  'Forcing_Analysis/geostrophic_test/'
run_folder = 'Re_big/'
nx = 512  # fourier resolution
nz = 68  # chebyshev resolution


H = 200  # depth of water in meters
h_e = 20 #ekman thickness
dh = h_e/H  # dh = ekman thickness divided by H
da = 0.01  # aspect ratio = ratio of height to length
f = 1e-4  # coriolis param in 1/s
L_func = lambda H, delta_a: H / delta_a
L = L_func(H, da)

def Rossby(R_e, R_g):

    R_e_temp = R_e / (2 * np.pi)
    R_g_temp = R_g / (2 * np.pi)

    nu_func = lambda f, H, delta_h: (f * (H ** 2) * (delta_h) ** 2) / 2
    tau_func = lambda f, delta_h, Re, delta_a, H: (Re * f ** 2) * (H ** 2) * (delta_h / delta_a)
    r_func = lambda f, delta_h, Re, Rg: (delta_h * Re * f) / (Rg)

    nu_value_temp = nu_func(f, H, dh)  # m^2/second
    tau_value_temp = tau_func(f, dh, R_e_temp, da, H)
    r_value_temp = r_func(f, dh, R_e_temp, R_g_temp)
    f_value_temp = f

    return nu_value_temp,f_value_temp,r_value_temp,tau_value_temp

ek_rossby = 1.75
geo_rossby = 0.1
A_v = Rossby(ek_rossby,geo_rossby)[0] #vertical viscosity
A_h =  Rossby(ek_rossby,geo_rossby)[0] * ((300/2)**2)/(h_e**2) #horizontal viscosity
f = Rossby(ek_rossby,geo_rossby)[1] #vertical viscosity
r = Rossby(ek_rossby,geo_rossby)[2]
print("---------------------------------------------------------------------")
print("---------------------------------------------------------------------")
print("-------------------- PARAMETERS USED -------------------------")
print("nu = ", Rossby(ek_rossby, geo_rossby)[0])
print("f = ", Rossby(ek_rossby, geo_rossby)[1])
print("r = ", Rossby(ek_rossby, geo_rossby)[2])
print("tau = ", Rossby(ek_rossby, geo_rossby)[3])
print("L = ", L)
print("H = ", H)
print("h_e = ", h_e)
print("nx = ", nx)

########################################################################################################################


zcoord = d3.Coordinate('z')
dist = d3.Distributor(zcoord)
z_basis = d3.Chebyshev(zcoord, size=nz, bounds=(0, H))


psi_arr = np.zeros((N + 1, nx, nz))
psi_arr_corrected = np.zeros((N + 1, nx, nz))
psi_x_arr = np.zeros((N + 1, nx, nz))
psi_x_arr_corrected = np.zeros((N + 1, nx, nz))
psi_z_arr = np.zeros((N + 1, nx, nz))
psi_z_arr_corrected = np.zeros((N + 1, nx, nz))
zeta_arr = np.zeros((N + 1, nx, nz))
zeta_arr_corrected = np.zeros((N + 1, nx, nz))
zeta_x_arr = np.zeros((N + 1, nx, nz))
zeta_xx_arr = np.zeros((N + 1, nx, nz))
zeta_z_arr = np.zeros((N + 1, nx, nz))
zeta_zz_arr = np.zeros((N + 1, nx, nz))
zeta_zz_arr_corrected = np.zeros((N + 1, nx, nz))
zeta_xx_arr_corrected = np.zeros((N + 1, nx, nz))
v_z_arr = np.zeros((N + 1, nx, nz))
v_x_arr = np.zeros((N + 1, nx, nz))
v_arr = np.zeros((N + 1, nx, nz))
v_arr_corrected = np.zeros((N + 1, nx, nz))
v_x_arr_corrected= np.zeros((N + 1, nx, nz))
v_z_arr_corrected= np.zeros((N + 1, nx, nz))
max_vals = np.zeros(N + 1)



for i in range(N+1):
    folder_n = run_folder + 'out_' + str(i) + '_n'
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
        zeta_zz_arr[i, :, :] = np.array(file['tasks']['<zetazz>'])  # d/dz (zeta)
        zeta_xx_arr[i, :, :] = np.array(file['tasks']['<zetaxx>'])  # d/dz (zeta)
        v_x_arr[i, :, :] = np.array(file['tasks']['<vx>'])  #
        v_z_arr[i, :, :] = np.array(file['tasks']['<vz>'])  #
        v_arr[i, :, :] = np.array(file['tasks']['<v>'])  #
    if i == 0:
        psi_arr_corrected[i, :, :] = psi_arr[0, :, :]
        zeta_arr_corrected[i, :, :] = zeta_arr[0, :, :]
        zeta_zz_arr_corrected[i, :, :] = zeta_zz_arr[0, :, :]
        zeta_xx_arr_corrected[i, :, :] = zeta_xx_arr[0, :, :]
        psi_x_arr_corrected[i, :, :] = psi_x_arr[0, :, :]
        psi_z_arr_corrected[i, :, :] = psi_z_arr[0, :, :]
        v_arr_corrected[i, :, :] = v_arr[0, :, :]
        v_x_arr_corrected[i, :, :] = v_x_arr[0, :, :]
        v_z_arr_corrected[i, :, :] = v_z_arr[0, :, :]
    else:
        psi_arr_corrected[i, :, :] = psi_arr_corrected[i - 1, :, :] + psi_arr[i, :, :]
        zeta_arr_corrected[i, :, :] = zeta_arr_corrected[i - 1, :, :] + zeta_arr[i, :, :]
        zeta_zz_arr_corrected[i, :, :] = zeta_zz_arr_corrected[i - 1, :, :] + zeta_zz_arr[i, :, :]
        zeta_xx_arr_corrected[i, :, :] = zeta_xx_arr_corrected[i - 1, :, :] + zeta_xx_arr[i, :, :]
        psi_x_arr_corrected[i, :, :] = psi_x_arr_corrected[i - 1, :, :] + psi_x_arr[i, :, :]
        psi_z_arr_corrected[i, :, :] = psi_z_arr_corrected[i - 1, :, :] + psi_z_arr[i, :, :]
        v_arr_corrected[i, :, :] = v_arr_corrected[i - 1, :, :] + v_arr[i, :, :]
        v_x_arr_corrected[i, :, :] =v_x_arr_corrected[i - 1, :, :] + v_x_arr[i, :, :]
        v_z_arr_corrected[i, :, :] = v_z_arr_corrected[i - 1, :, :] + v_z_arr[i, :, :]

x = np.linspace(0, L, nx)
X, Z = np.meshgrid(dist.local_grid(z_basis),x )


w = lambda i: psi_x_arr_corrected[i, :, :]
u_nl = lambda i: psi_z_arr_corrected[i, :, :] - psi_z_arr_corrected[0,:,:]
vz_nl = lambda i: v_z_arr_corrected[i, :, :] - v_z_arr_corrected[0,:,:]
zeta_zz_nl = lambda i: zeta_zz_arr_corrected[i, :, :] - zeta_zz_arr_corrected[0,:,:]
zeta_xx_nl = lambda i: zeta_xx_arr_corrected[i, :, :] - zeta_xx_arr_corrected[0,:,:]
vmax = 0.003

order = 0
fig,ax= plt.subplots(constrained_layout=True)
CS = plt.contour(Z, X, psi_arr_corrected[0, :, :], 30, colors='k')
plt.clabel(CS, CS.levels[1::5],inline=1,fontsize=5)
CM= plt.pcolormesh(Z, X, w(0), shading='gouraud',cmap='PRGn', vmin=-vmax,vmax=vmax)
cbar = fig.colorbar(CM)
cbar.ax.set_ylabel('Velocity Field')
plt.ylabel('vertical depth')
plt.xlabel('periodic x-axis (0,2$\pi$)')
plt.title('$w^0$')
plt.savefig(run_folder + 'w_o'+ str(order)+ '.png')
plt.close(fig)

order = 1
fig,ax= plt.subplots(constrained_layout=True)
CS = plt.contour(Z, X, psi_arr_corrected[1, :, :], 30, colors='k')
plt.clabel(CS, CS.levels[1::5],inline=1,fontsize=5)
CM= plt.pcolormesh(Z, X, w(1), shading='gouraud',cmap='PRGn', vmin=-vmax,vmax=vmax)
cbar = fig.colorbar(CM)
cbar.ax.set_ylabel('Velocity Field')
plt.ylabel('vertical depth')
plt.xlabel('periodic x-axis (0,2$\pi$)')
plt.title('$w^0 + w^1$ Colormap')
plt.savefig(run_folder + 'w_o'+ str(order)+ '.png')
plt.close(fig)

order = 2
fig,ax= plt.subplots(constrained_layout=True)
CS = plt.contour(Z, X, psi_arr_corrected[2, :, :], 30, colors='k')
plt.clabel(CS, CS.levels[1::5],inline=1,fontsize=5)
CM= plt.pcolormesh(Z, X, w(2), shading='gouraud',cmap='PRGn', vmin=-vmax,vmax=vmax)
cbar = fig.colorbar(CM)
cbar.ax.set_ylabel('Velocity Field')
plt.ylabel('vertical depth')
plt.xlabel('periodic x-axis (0,2$\pi$)')
plt.title('$w^0 + w^1 + w^2$ Colormap')
plt.savefig(run_folder + 'w_o'+ str(order)+ '.png')
plt.close(fig)

order = N
fig,ax= plt.subplots(constrained_layout=True)
CS = plt.contour(Z, X, psi_arr_corrected[N, :, :], 30, colors='k')
plt.clabel(CS, CS.levels[1::5],inline=1,fontsize=5)
CM= plt.pcolormesh(Z, X, w(order), shading='gouraud',cmap='PRGn', vmin=-vmax,vmax=vmax)
cbar = fig.colorbar(CM)
cbar.ax.set_ylabel('Velocity Field')
plt.ylabel('vertical depth')
plt.xlabel('periodic x-axis (0,2$\pi$)')
plt.title('$\sum_{n=0}^{100} w^{n}$ Colormap')
plt.savefig(run_folder + 'w_o'+ str(order)+ '.png')
plt.close(fig)

#######################################################

order = N

fig,ax= plt.subplots(constrained_layout=True)
CS = plt.contour(Z, X, psi_arr_corrected[order, :, :], 30, colors='k')
plt.clabel(CS, CS.levels[1::5],inline=1,fontsize=5)
#CM= plt.pcolormesh(Z, X, A_v * zeta_zz_nl(order), shading='gouraud',cmap='PRGn')
CM= plt.pcolormesh(Z, X, A_v * zeta_zz_nl(order), shading='gouraud',cmap='PRGn')
cbar = fig.colorbar(CM)
cbar.ax.set_ylabel('Field Strength')
plt.ylabel('vertical depth')
plt.xlabel('periodic x-axis (0,2$\pi$)')
plt.title('$A_v(\sum_{n=0}^{100} \zeta_{zz}^n - \zeta_{zz}^0)$ Colormap')
plt.savefig(run_folder +'Azeta_zz_o'+ str(order)+ '.png')
plt.close(fig)

fig,ax= plt.subplots(constrained_layout=True)
CS = plt.contour(Z, X, psi_arr_corrected[order, :, :], 30, colors='k')
plt.clabel(CS, CS.levels[1::5],inline=1,fontsize=5)
#CM= plt.pcolormesh(Z, X, A_v * zeta_zz_nl(order), shading='gouraud',cmap='PRGn')
CM= plt.pcolormesh(Z, X, A_h * zeta_xx_nl(order), shading='gouraud',cmap='PRGn')
cbar = fig.colorbar(CM)
cbar.ax.set_ylabel('Field Strength')
plt.ylabel('vertical depth')
plt.xlabel('periodic x-axis (0,2$\pi$)')
plt.title('$A_h(\sum_{n=0}^{100} \zeta_{xx}^n - \zeta_{xx}^0)$ Colormap')
plt.savefig(run_folder +'Azeta_xx_o'+ str(order)+ '.png')
plt.close(fig)

fig,ax= plt.subplots(constrained_layout=True)
CS = plt.contour(Z, X, psi_arr_corrected[order, :, :], 30, colors='k')
plt.clabel(CS, CS.levels[1::5],inline=1,fontsize=5)
CM= plt.pcolormesh(Z, X, f* vz_nl(order), shading='gouraud',cmap='PRGn')
cbar = fig.colorbar(CM)
cbar.ax.set_ylabel('Field Strength')
plt.ylabel('vertical depth')
plt.xlabel('periodic x-axis (0,2$\pi$)')
plt.title('$f(\sum_{n=0}^{100} v^n - v_z^0)$ Colormap')
plt.savefig(run_folder +'fv_z_o'+ str(order)+ '.png')
plt.close(fig)

fig,ax= plt.subplots(constrained_layout=True)
CS = plt.contour(Z, X, psi_arr_corrected[0, :, :], 30, colors='k')
plt.clabel(CS, CS.levels[1::5],inline=1,fontsize=5)
CM= plt.pcolormesh(Z, X, A_v*zeta_zz_arr_corrected[0,:,:], shading='gouraud',cmap='PRGn')
cbar = fig.colorbar(CM)
cbar.ax.set_ylabel('Field Strength')
plt.ylabel('vertical depth')
plt.xlabel('periodic x-axis (0,2$\pi$)')
plt.title('$\zeta_{zz}^0$ Colormap')
plt.savefig(run_folder +'zeta_zz_o'+ str(order)+ '.png')
plt.close(fig)

fig,ax= plt.subplots(constrained_layout=True)
CS = plt.contour(Z, X, psi_arr_corrected[0, :, :], 30, colors='k')
plt.clabel(CS, CS.levels[1::5],inline=1,fontsize=5)
CM= plt.pcolormesh(Z, X, f* v_z_arr_corrected[0,:,:], shading='gouraud',cmap='PRGn')
cbar = fig.colorbar(CM)
cbar.ax.set_ylabel('Field Strength')
plt.ylabel('vertical depth')
plt.xlabel('periodic x-axis (0,2$\pi$)')
plt.title('$\sum_{n=0}^{100} v_z^n$ Colormap')
plt.savefig(run_folder +'v_z_o'+ str(order)+ '.png')
plt.close(fig)

fig,ax= plt.subplots(constrained_layout=True)
CS = plt.contour(Z, X, psi_arr_corrected[order, :, :], 30, colors='k')
plt.clabel(CS, CS.levels[1::5],inline=1,fontsize=5)
CM= plt.pcolormesh(Z, X, zeta_arr_corrected[order,:,:], shading='gouraud',cmap='PRGn')
cbar = fig.colorbar(CM)
cbar.ax.set_ylabel('Field Strength')
plt.ylabel('vertical depth')
plt.xlabel('periodic x-axis (0,2$\pi$)')
plt.title('$\sum_{n=0}^{100} \zeta^n$ Colormap')
plt.savefig(run_folder +'zeta_o'+ str(order)+ '.png')
plt.close(fig)

fig,ax= plt.subplots(constrained_layout=True)
CS = plt.contour(Z, X, psi_arr_corrected[order, :, :], 30, colors='k')
plt.clabel(CS, CS.levels[1::5],inline=1,fontsize=5)
CM= plt.pcolormesh(Z, X, zeta_arr_corrected[order,:,:]- zeta_arr_corrected[0,:,:], shading='gouraud',cmap='PRGn')
cbar = fig.colorbar(CM)
cbar.ax.set_ylabel('Field Strength')
plt.ylabel('vertical depth')
plt.xlabel('periodic x-axis (0,2$\pi$)')
plt.title('$\sum_{n=0}^{100} \zeta^n - \zeta^0$ Colormap')
plt.savefig(run_folder +'zetadifff_o'+ str(order)+ '.png')
plt.close(fig)

fig,ax= plt.subplots(constrained_layout=True)
CS = plt.contour(Z, X, psi_arr_corrected[order, :, :], 30, colors='k')
plt.clabel(CS, CS.levels[1::5],inline=1,fontsize=5)
CM= plt.pcolormesh(Z, X, (f* vz_nl(order)) +A_v * zeta_zz_nl(order) , shading='gouraud',cmap='PRGn')
cbar = fig.colorbar(CM)
cbar.ax.set_ylabel('Field Strength')
plt.ylabel('vertical depth')
plt.xlabel('periodic x-axis (0,2$\pi$)')
plt.title('$\sum_{n=0}^{100} J^n-J^0$ Colormap')
plt.savefig(run_folder +'Jdiff_o'+ str(order)+ '.png')
plt.close(fig)