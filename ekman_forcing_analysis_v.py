"""

This file is the solver for diagnosing which terms contribute to the Ekman asymmetry we have found. In particular it is
for splitting up y-velocity v into v_A, v_B such that:

 grad^2 v = grad^2 v_A + grad^2 v_B
 grad^2 v_A = fu
 grad^2 v_B = J(psi,v)

Solving for v_A, v_B tells us what the vorticity would look like if only forced by the Jacobian or the dissipative
term respectively, ultimately helping find terms which dominantly contribute to the Ekman asymmetry

"""

import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
import sys
from scipy.fft import fft, fftfreq
from dedalus import public as d3

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
zeta_arr = np.zeros((N + 1, nx, nz))
zeta_arr_corrected = np.zeros((N + 1, nx, nz))
v_z_arr = np.zeros((N + 1, nx, nz))
v_z_arr_corrected = np.zeros((N + 1, nx, nz))
u_arr= np.zeros((N + 1, nx, nz))
u_arr_corrected= np.zeros((N + 1, nx, nz))

# folder_n = run_folder + 'out_' + str(i) + '_n'
# folder_n_sub = 'out_' + str(self.n) + '_n'
########################################################################################################################
for i in range(N+1):
    folder_n = run_folder +'out_' + str(i) + '_n'
    folder_n_sub = 'out_' + str(i) + '_n'
    with h5py.File(folder_n + '/' + folder_n_sub + '_s1/' + folder_n_sub + '_s1_p0.h5',
                   mode='r') as file:  # reading file
        u_arr[i, :, :] = np.array(file['tasks']['<u>'])
        psi_arr[i, :, :] = np.array(file['tasks']['<psi>'])  # psi
        zeta_arr[i, :, :] = np.array(file['tasks']['<zeta>'])  # zeta
        v_z_arr[i, :, :] = np.array(file['tasks']['<vz>'])  #
        J_v_arr[i, :, :] = np.array(file['tasks']['<J_psi_v>'])
        J_zeta_arr[i, :, :] = np.array(file['tasks']['<J>'])
    if i == 0:
        u_arr_corrected[i, :, :] = u_arr[0, :, :]
        psi_arr_corrected[i, :, :] = psi_arr[0, :, :]
        zeta_arr_corrected[i, :, :] = zeta_arr[0, :, :]
        v_z_arr[i, :, :] = v_z_arr[0,:,:]
        J_v_arr_corrected[i, :, :] = J_v_arr[0, :, :]
        J_zeta_arr_corrected[i, :, :] = J_zeta_arr[0, :, :]
    else:
        u_arr_corrected[i, :, :] = u_arr_corrected[i - 1, :, :] + u_arr[i,:,:]
        psi_arr_corrected[i, :, :] = psi_arr_corrected[i - 1, :, :] + psi_arr[i, :, :]
        zeta_arr_corrected[i, :, :] = zeta_arr_corrected[i - 1, :, :] + zeta_arr[i, :, :]
        v_z_arr_corrected[i, :, :] = v_z_arr_corrected[i - 1, :, :] + v_z_arr[i, :, :]
        J_v_arr_corrected[i, :, :] = J_v_arr_corrected[i-1, :, :] + J_v_arr[i, :, :]
        J_zeta_arr_corrected[i, :, :] = J_zeta_arr_corrected[i-1, :, :] + J_zeta_arr[i, :, :]

x = np.linspace(0, L, nx)
z = np.linspace(0, H, nz)

x_basis = d3.Fourier('x', nx, interval=(0, L))
z_basis = d3.Chebyshev('z', nz, interval=(0, H))
domain = d3.Domain([x_basis, z_basis], grid_dtype=np.float64)

problem = d3.LBVP(domain, variables=['psi','psi_z' 'v_A',
                                             'v_Az','v_Azz', 'v_B',
                                             'v_Bz','v_Bzz'])
a_visc = ((300 / 2) ** 2) / (1** 2)
# setting up all parameters
problem.parameters['nu'] = 5*1e-5  # viscosity
problem.parameters['nu_h'] = a_visc * 5*1e-5
problem.parameters['f'] = 1e-4 # coriolis param


Jac_temp_v = domain.new_field()
gslices = domain.dist.grid_layout.slices(scales=1)
Jac_temp_v['g'] = J_v_arr_corrected[100,:,:][gslices[0]]

Jac_temp_zeta = domain.new_field()
gslices = domain.dist.grid_layout.slices(scales=1)
Jac_temp_zeta['g'] = J_zeta_arr_corrected[100,:,:][gslices[0]]-J_zeta_arr_corrected[0,:,:][gslices[0]]

u_temp = domain.new_field()
gslices = domain.dist.grid_layout.slices(scales=1)
u_temp['g'] = u_arr_corrected[100,:,:][gslices[0]]-u_arr_corrected[0,:,:][gslices[0]]

#problem.parameters['Jac_v'] = Jac_temp_v
problem.parameters['Jac_v'] = Jac_temp_v
problem.parameters['u'] = u_temp


problem.add_equation("v_Az - dz(v_A)=0")  # auxilary
problem.add_equation("v_Azz - dz(v_Az)=0")  # auxilary
problem.add_equation("psi_Az - dz(psi_A)=0")  # auxilary

problem.add_equation("zeta_Bz - dz(zeta_B)=0")  # auxilary
problem.add_equation("zeta_Bzz - dz(zeta_Bz)=0")  # auxilary
problem.add_equation("psi_Bz - dz(psi_B)=0")  # auxilary


problem.add_equation("zeta - dz(psi_z) - dx(dx(psi))=0")

problem.add_equation("(dx(dx(v_A))*nu_h + v_Azz*nu) =  f*u")  # nu* grad^2 zeta + fv_z=0
problem.add_equation("(dx(dx(v_B))*nu_h + v_Bzz*nu) = Jac_v")


problem.add_bc("psi(z='left') = 0")
problem.add_bc("psi(z='right') = 0")
problem.add_bc("dz(psi_z)(z='left') = 0")
problem.add_bc("dz(psi_z)(z='left') = 0")


solver = problem.build_solver()
solver.solve()
state = solver.state['psi_A']

#output:
folder = 'Forcing_Analysis/'
folder_n = 'Forcing_Analysis/' + 'out'
folder_n_sub = 'out'

out = solver.evaluator.add_file_handler(folder_n)  # storing output into file with specified name
out.add_system(solver.state)

out.add_task("psi_A", layout='g', name='<psi_A>')  # saving variables
out.add_task("zeta_A", layout='g', name='<zeta_A>')  # saving variables
out.add_task("psi_B", layout='g', name='<psi_B>')  # saving variables
out.add_task("zeta_B", layout='g', name='<zeta_B>')  # saving variables
solver.evaluator.evaluate_handlers([out], world_time=0, wall_time=0, sim_time=0, timestep=0, iteration=0)
