"""
Dedalus script for 2D Rayleigh-Benard convection.
This script uses a Fourier basis in the x direction with periodic boundary
conditions.  The equations are scaled in units of the buoyancy time (Fr = 1).
This script can be ran serially or in parallel, and uses the built-in analysis
framework to save data snapshots in HDF5 files.  The `merge_procs` command can
be used to merge distributed analysis sets from parallel runs, and the
`plot_slices.py` script can be used to plot the snapshots.
To run, merge, and plot using 4 processes, for instance, you could use:
    $ mpiexec -n 2 order_n_time_dependent_d2.py
    $ mpiexec -n 2 python3 -m dedalus merge_procs snapshots
    $ mpiexec -n 2 python3 plot_slices.py snapshots/*.h5
This script can restart the simulation from the last save of the original
output to extend the integration.  This requires that the output files from
the original simulation are merged, and the last is symlinked or copied to
`restart.h5`.
To run the original example and the restart, you could use:
    $ mpiexec -n 4 python3 rayleigh_benard.py
    $ mpiexec -n 4 python3 -m dedalus merge_procs snapshots
    $ ln -s snapshots/snapshots_s2.h5 restart.h5
    $ mpiexec -n 4 python3 rayleigh_benard.py
The simulations should take a few process-minutes to run.
"""

import numpy as np
from mpi4py import MPI
import time
import pathlib

from dedalus import public as de
from dedalus.extras import flow_tools

import logging
logger = logging.getLogger(__name__)


# Parameters
nx = 512 # fourier resolution
nz = 128  # chebyshev resolution

H = 100  # depth of water in meters
h_e = 10 #ekman thickness
dh = h_e/H  # dh = ekman thickness divided by H
da = 0.01  # aspect ratio = ratio of height to length
f = 1e-4  # coriolis param in 1/s

L_func = lambda H, delta_a: H / delta_a
L = L_func(H, da)
k = (2 * np.pi) / (L)

Re = 1.6
Rg=0.1

print(L)
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

    # print('L=', L)
    # print('tau=', tau_value_temp)
    # print('r=', r_value_temp)
    # print('nu=', nu_value_temp)

    return nu_value_temp,f_value_temp,r_value_temp,tau_value_temp

print(Rossby(Re,Rg))

x_basis = de.Fourier('x', nx, interval=(0, L))
z_basis = de.Chebyshev('z', nz, interval=(0, H))
domain = de.Domain([x_basis, z_basis], grid_dtype=np.float64)


problem = de.IVP(domain, variables=['psi', 'u', 'v', 'vx',
                                     'vz', 'zeta',
                                     'zetaz', 'zetax', 'psix', 'psiz', 'zetazz'])

a_visc = ((300 / 2) ** 2) / (h_e ** 2)
# setting up all parameters
problem.parameters['nu'] = Rossby(Re,Rg)[0]  # viscosity
problem.parameters['nu_h'] = a_visc *  Rossby(Re,Rg)[0]
problem.parameters['f'] = Rossby(Re,Rg)[1]   # coriolis param
problem.parameters['r'] = Rossby(Re,Rg)[2]   # damping param
problem.parameters['A'] = Rossby(Re,Rg)[3]   # forcing param
problem.parameters['H'] = H
problem.parameters['k'] = k

# auxliary equations:
problem.add_equation("u - psiz = 0")
problem.add_equation("vz - dz(v) = 0")  # auxilary
problem.add_equation("vx - dx(v) = 0")  # auxilary
problem.add_equation("zetaz - dz(zeta)=0")  # auxilary
problem.add_equation("zetazz - dz(zetaz)=0")  # auxilary
problem.add_equation("zetax - dx(zeta)=0")  # auxilary
problem.add_equation("psiz - dz(psi)=0")  # auxilary
problem.add_equation("psix - dx(psi)=0")  # auxilary
problem.add_equation("zeta - dz(u) - dx(dx(psi))=0")  # zeta = grad^2(psi)

problem.add_equation(" dt(v) - (dx(dx(v))*nu_h + dz(vz)*nu) - r*(1/H)*integ(v,'z')  +f*u =0")
problem.add_equation(" dt(u) - (dx(dx(zeta))*nu_h + zetazz*nu) - f*vz = 0 ")

# Boundary conditions:
problem.add_bc("psi(z='left') = 0")
problem.add_bc("psi(z='right') = 0")
problem.add_bc("vz(z='left') = 0")
problem.add_bc("vz(z='right') = (A/nu)*cos(x*k+ 3.14159/2)")
problem.add_bc("dz(u)(z='left') = 0")
problem.add_bc("dz(u)(z='right') = 0")

# Build solver
solver = problem.build_solver(de.timesteppers.RK222)
logger.info('Solver built')

# Initial conditions
x, z = domain.all_grids()
psi = solver.state['psi']
zeta = solver.state['zeta']



# Timestepping and output
dt = 1e6
stop_sim_time = 1e8
fh_mode = 'overwrite'


# Integration parameters
solver.stop_sim_time = stop_sim_time

# Analysis
#snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=0.25, max_writes=50, mode=fh_mode)
snapshots = solver.evaluator.add_file_handler('Re_big/ivp/linear/snapshots', iter=10, max_writes=150)
snapshots.add_system(solver.state)
snapshots.add_task("psi", layout='g', name='<psi>')
snapshots.add_task("v", layout='g', name='<v>') # saving variables
#solver.evaluator.evaluate_handlers([snapshots], world_time=0, wall_time=0, sim_time=solver.sim_time, timestep=dt, iteration=stop_sim_time/dt)

# CFL
CFL = flow_tools.CFL(solver, initial_dt=dt, cadence=10, safety=1,
                     max_change=1.5, min_change=0.5, max_dt=dt+10, min_dt=dt-10, threshold=0.05)
CFL.add_velocities(('u', 'v'))

# Flow properties
flow = flow_tools.GlobalFlowProperty(solver, cadence=10)
flow.add_property("sqrt(u*u)", name='U')

# Main loop
try:
    logger.info('Starting loop')
    while solver.proceed:
        dt = CFL.compute_dt()
        dt = solver.step(dt)
        if (solver.iteration-1) % 10 == 0:
            logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))
            logger.info(' U = %f' %flow.max('U'))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
# finally:
#     solver.log_stats()

