import os
import numpy as np
from tqdm import tqdm
from tqdm.notebook import tqdm_notebook
import time
import h5py
import math
import matplotlib.pyplot as plt
import sys
from contextlib import suppress

# sys.path[0] = ""
from dedalus import public as d3
from dedalus.extras import plot_tools

d3.logging_setup.rootlogger.setLevel('ERROR')  # suppress logging msgs

nx = 512  # fourier resolution
nz = 128  # chebyshev resolution

H = 10  # depth of water in meters
h_e = 1  # ekman thickness

dh = h_e / H  # dh = ekman thickness divided by H
da = 0.01  # aspect ratio = ratio of height to length
f = 1e-4  # coriolis param in 1/s

N = 1  # the order up to which the model runs
stop_sim_time = 2
L_func = lambda H, delta_a: H / delta_a
L = L_func(H, da)
print("L = ", L)
k = (2 * np.pi) / (L)

# Initializing arrays to store values at each order (adding one to the order column to account for indexing limits)
psi_arr = np.zeros((N + 1, nx, nz))
psi_arr_corrected = np.zeros((N + 1, nx, nz))
psi_x_arr = np.zeros((N + 1, nx, nz))
psi_z_arr = np.zeros((N + 1, nx, nz))
zeta_arr = np.zeros((N + 1, nx, nz))
zeta_x_arr = np.zeros((N + 1, nx, nz))
zeta_xx_arr = np.zeros((N + 1, nx, nz))
zeta_z_arr = np.zeros((N + 1, nx, nz))
zeta_zz_arr = np.zeros((N + 1, nx, nz))
v_z_arr = np.zeros((N + 1, nx, nz))
v_x_arr = np.zeros((N + 1, nx, nz))
v_arr = np.zeros((N + 1, nx, nz))
max_vals = np.zeros(N + 1)

run_folder = 'Re_timed/'

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

    return nu_value_temp, f_value_temp, r_value_temp, tau_value_temp

# Bases
dealias = 3/2
coords = d3.CartesianCoordinates('x', 'z')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=nx, bounds=(0, L), dealias=dealias)
zbasis = d3.RealFourier(coords['z'], size=nz, bounds=(0, H), dealias=dealias)

# Fields
psi = dist.Field(name='psi', bases=(xbasis,zbasis))
zeta = dist.Field(name='zeta', bases=(xbasis,zbasis))
v = dist.VectorField(coords, name='v', bases=(xbasis,zbasis))
u = dist.VectorField(coords, name='u', bases=(xbasis,zbasis))
tau_psi = dist.Field(name='tau_psi')

# Problem
problem = d3.IVP([psi,u,v,zeta,tau_psi], namespace=locals())

a_visc = ((300 / 2) ** 2) / (h_e ** 2)

# setting up all parameters
problem.parameters['nu'] = self.nu  # viscosity
problem.parameters['nu_h'] = a_visc * self.nu
problem.parameters['f'] = self.f  # coriolis param
problem.parameters['r'] = self.r  # damping param
problem.parameters['A'] = self.tau  # forcing param
problem.parameters['H'] = H
problem.parameters['k'] = k

#problem.parameters['Jac'] = self.Jn(self.n)[0]  # self.j = self.j2 = 0 for 0th order solution
#problem.parameters['Jac_psi_v'] = self.Jn(self.n)[1]

# equations:
problem.add_equation("u - psiz = 0")
problem.add_equation("zeta - Lap(psi)=0")  # zeta = grad^2(psi)

problem.add_equation(
    "dt(u) + Lap(v)*nu - r*(1/H)*integ(v,'z')  -f*u = 0")  # nu* grad^2 v  - fu=0
problem.add_equation("dt(v) + Lap(zeta)*nu) + f*dz(v) = 0")  # nu* grad^2 zeta + fv_z=0


problem.add_bc("vz(z='right') = (A/nu)*cos(x*k+ 3.14159/2)")  # wind forcing on 0th order boundary condition


# Boundary conditions:
problem.add_bc("psi(z='left') = 0")
problem.add_bc("psi(z='right') = 0")
problem.add_bc("vz(z='left') = 0")
problem.add_bc("dz(u)(z='left') = 0")
problem.add_bc("dz(u)(z='right') = 0")

# Building Solver:
solver = problem.build_solver(d3.RK222)
solver.stop_sim_time = stop_sim_time
solver.solve()
