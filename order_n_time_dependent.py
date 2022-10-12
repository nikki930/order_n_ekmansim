import os
import numpy as np

import time
import h5py
import math
import matplotlib.pyplot as plt
import sys
from contextlib import suppress

# sys.path[0] = ""
from dedalus import public as d3
from dedalus.extras import plot_tools

R_e=1.6
R_g=0.1

nx = 512  # fourier resolution
nz = 128  # chebyshev resolution

H = 10  # depth of water in meters
h_e = 1  # ekman thickness

dh = h_e / H  # dh = ekman thickness divided by H
da = 0.01  # aspect ratio = ratio of height to length
f = 1e-4  # coriolis param in 1/s

N = 1  # the order up to which the model runs
dtype=np.float64
stop_sim_time = 2
L_func = lambda H, delta_a: H / delta_a
L = L_func(H, da)
print("L = ", L)
k = (2 * np.pi) / (L)


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
dealias = 1
coords = d3.CartesianCoordinates('x', 'z')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=nx, bounds=(0, L), dealias=dealias)
zbasis = d3.ChebyshevT(coords['z'], size=nz, bounds=(0, H), dealias=dealias)

# Fields
psi = dist.Field(name='psi', bases=(xbasis,zbasis))
zeta = dist.Field(name='zeta', bases=(xbasis,zbasis))
v = dist.Field(name='v', bases=(xbasis,zbasis))
u = dist.Field(name='u', bases=(xbasis,zbasis))
tau_psi1 = dist.Field(name='tau_psi1', bases=xbasis)
tau_psi2 = dist.Field(name='tau_psi2', bases=xbasis)
tau_u1 = dist.Field(name='tau_u1', bases=xbasis)
tau_u2 = dist.Field(name='tau_u2', bases=xbasis)
tau_v1 = dist.Field(name='tau_v1', bases=xbasis)
tau_v2 = dist.Field(name='tau_v2', bases=xbasis)


# Problem
problem = d3.IVP([psi,u,v,tau_psi1,tau_psi2,tau_u1,tau_u2,tau_v1,tau_v2], namespace=locals())

a_visc = ((300 / 2) ** 2) / (h_e ** 2)

# setting up all parameters
nu= Rossby(R_e,R_g)[0]  # viscosity
nu_h= a_visc * Rossby(R_e,R_g)[0]
f = Rossby(R_e,R_g)[1]  # coriolis param
r = Rossby(R_e,R_g)[2]  # damping param
A = Rossby(R_e,R_g)[3] # forcing param

x, z = dist.local_grids(xbasis, zbasis)
ex, ez = coords.unit_vector_fields(dist)

#substitutions:
dx = lambda A: d3.Differentiate(A,coords[0])
dz = lambda A: d3.Differentiate(A,coords[1])
integrate = lambda A: d3.Integrate(d3.Integrate(A, 'z'))

lift_basis = zbasis.derivative_basis(1)
lift = lambda A: d3.Lift(A, lift_basis, -1)
# lift_basis1 = zbasis.derivative_basis(1) #lift to first derivative basis
# lift1= lambda A,n: d3.Differentiate(A,lift_basis1,n)

grad_psi = d3.grad(psi) + ez*lift(tau_psi1) # First-order reduction
grad_v=d3.grad(v) + ez*lift(tau_v1) # First-order reduction
zeta = d3.div(grad_psi) + lift(tau_psi1) # First-order reduction
grad_zeta= d3.grad(zeta)+ ez*lift(tau_psi1)

forcing = dist.Field(bases=(xbasis,zbasis))
forcing['g']=np.cos(x*k+ 3.14159/2)

# equations:
# First-order form: "lap(f)" becomes "div(grad_f)"
problem.add_equation("u - dz(psi) + tau_u1 +tau_u2+ lift(tau_psi1) +lift(tau_psi2) = 0")
#problem.add_equation("zeta - div(grad(psi))+ lift1(tau_psi1,-1) +lift1(tau_psi2,-2) =0")  # zeta = grad^2(psi)
problem.add_equation( "dt(u) + div(grad_v)*nu - r*(1/H)*integrate(v)  -f*u  +lift(tau_v2) + tau_u1 + tau_u2 = 0")  # nu* grad^2 v  - fu=0
problem.add_equation("dt(v) + div(grad_zeta)*nu + f*dz(v) +lift(tau_psi2) + lift(tau_v1) +lift(tau_v2) = 0")  # nu* grad^2 zeta + fv_z=0

problem.add_equation("dz(u)(z=0)=0")
problem.add_equation("dz(u)(z=H)=0")
problem.add_equation("psi(z=0)=0")
problem.add_equation("psi(z=H)=0")
problem.add_equation("dz(v)(z=0)=0")
problem.add_equation("dz(u)(z=H)=(A/nu)*forcing")


# Building Solver:
solver = problem.build_solver(d3.RK222)
solver.stop_sim_time = stop_sim_time
solver.solve()
