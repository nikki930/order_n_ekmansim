
"""
2D Ekman Solver from Navier-Stokes equations
"""

from mpi4py import MPI
import numpy as np
from tqdm import tqdm
from tqdm.notebook import tqdm_notebook
import time
import h5py
import matplotlib.pyplot as plt



from dedalus import public as de
from dedalus.extras import flow_tools
de.logging_setup.rootlogger.setLevel('ERROR')  # suppress logging msgs
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

N = 10 # the order up to which the model runs




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

    return nu_value_temp, f_value_temp, r_value_temp, tau_value_temp


x_basis = de.Fourier('x', nx, interval=(0, L))
z_basis = de.Chebyshev('z', nz, interval=(0, H))
domain = de.Domain([x_basis, z_basis], grid_dtype=np.float64)

# declaring domain and variables for the problem
problem = de.IVP(domain, variables=['psi', 'u', 'v', 'vx',
                                     'vz', 'zeta',
                                     'zetaz', 'zetax', 'psix', 'psiz', 'zetazz'])

n_time = 10 #size of time array

# Initializing arrays to store values at each order (adding one to the order column to account for indexing limits)
psi_arr = np.zeros((N + 1, n_time, nx, nz))
psi_arr_corrected = np.zeros((N + 1, n_time, nx, nz))
psi_x_arr = np.zeros((N + 1, n_time, nx, nz))
psi_z_arr = np.zeros((N + 1, n_time, nx, nz))
zeta_arr = np.zeros((N + 1, n_time, nx, nz))
zeta_x_arr = np.zeros((N + 1, n_time, nx, nz))
zeta_xx_arr = np.zeros((N + 1, n_time, nx, nz))
zeta_z_arr = np.zeros((N + 1, n_time, nx, nz))
zeta_zz_arr = np.zeros((N + 1, n_time, nx, nz))
v_z_arr = np.zeros((N + 1, n_time, nx, nz))
v_x_arr = np.zeros((N + 1, n_time, nx, nz))
v_arr = np.zeros((N + 1, n_time, nx, nz))
max_vals = np.zeros(N + 1)
t_arr= np.zeros((n_time))
run_folder = 'Re_big/ivp/'


class Solver_n:
    def __init__(self, visc, coriolis, damp, wind, order):

        '''
        Class
        :param visc:
        :param coriolis:
        :param damp:
        :param wind:
        :param order:
        '''

        global nu
        global r
        global tau

        self.nu = visc
        self.f = coriolis
        self.r = damp  # damping parameter to dampen as depth increases
        self.tau = wind
        self.n = order

        nu = self.nu
        r = self.r
        tau = self.tau

    def Jn(self, t_idx, n_solve):

        '''
        Calculates the nonlinear terms of order n

        :param self:
        :param t_idx: index of time at which
        :param n_solve: order to solve J to
        :return: [J(psi,zeta), J(psi,v)]
        '''
        J1 = 0
        J2 = 0
        folder0 = 'out_0_n'
        global psi_arr

        if n_solve == 0:
            return J1, J2

        else:
            n = n_solve - 1
            # print ("n_solve-1=",n)
            for j in range(0, n + 1):
                J1 += psi_x_arr[j, :, :,:] * zeta_z_arr[n - j, :, :,:] - psi_z_arr[j, :, :,:] * zeta_x_arr[n - j, :, :,:]
                J2 += psi_x_arr[j, :, :,:] * v_z_arr[n - j, :, :,:] - psi_z_arr[j, :, :,:] * v_x_arr[n - j, :, :,:]

            Jac_temp1 = domain.new_field()
            gslices = domain.dist.grid_layout.slices(scales=1)
            Jac_temp1['g'] = J1[t_idx,:,:][gslices[0]]

            Jac_temp2 = domain.new_field()
            gslices = domain.dist.grid_layout.slices(scales=1)
            Jac_temp2['g'] = J2[t_idx,:,:][gslices[0]]

            return Jac_temp1, Jac_temp2


    def eqns(self, state_var):

        '''
        Using Dedalus linear boundary value problem to build a solver, where in each step the nonlinear terms are
        calculated in Jn() and implemented as forcings to the flow.

        :param state_var: choice of state_variable desired to solve for
        :return:
        '''

        global solver
        global state
        global dt
        global Jac_psi_zeta
        global Jac_psi_v

        # def data_1(*args, domain=domain, F=self.Jn):
        #
        #     return de.operators.GeneralFunction(domain, layout='g', func=F(self.n)[0], args=args)
        #
        # def data_2(*args, domain=domain, F=self.Jn):
        #
        #     return de.operators.GeneralFunction(domain, layout='g', func=F(self.n)[1], args=args)
        if self.n==0:
            Jac_psi_zeta = 0
            Jac_psi_v = 0

        else:
            Jac_psi_zeta = self.Jn(0, self.n)[0]
            Jac_psi_v = self.Jn(0, self.n)[1]

        problem = de.IVP(domain, variables=['psi', 'u', 'v', 'vx',
                                            'vz', 'zeta',
                                            'zetaz', 'zetax', 'psix', 'psiz', 'zetazz'])

        a_visc = ((300 / 2) ** 2) / (h_e ** 2)
        # setting up all parameters
        problem.parameters['nu'] = self.nu  # viscosity
        problem.parameters['nu_h'] = a_visc * self.nu
        problem.parameters['f'] = self.f  # coriolis param
        problem.parameters['r'] = self.r  # damping param
        problem.parameters['A'] = self.tau  # forcing param
        problem.parameters['H'] = H
        problem.parameters['k'] = k


            #
            # Jac_psi_zeta.meta['x', 'z']['constant'] = True
            # Jac_psi_v.meta['x', 'z']['constant'] = True
            #de.operators.parseables['Jac_psi_v'] = data_2()
            #de.operators.parseables['Jac_psi_zeta'] =data_1()

        problem.parameters['Jac_psi_zeta'] = Jac_psi_zeta
        problem.parameters['Jac_psi_v'] =Jac_psi_v


        # de.operators.parseables['Jac_psi_v'] = data_2()
        # de.operators.parseables['Jac_psi_zeta'] =data_1()
        # axuliary equations:
        problem.add_equation("u - psiz = 0")
        problem.add_equation("vz - dz(v) = 0")  # auxilary
        problem.add_equation("vx - dx(v) = 0")  # auxilary
        problem.add_equation("zetaz - dz(zeta)=0")  # auxilary
        problem.add_equation("zetazz - dz(zetaz)=0")  # auxilary
        problem.add_equation("zetax - dx(zeta)=0")  # auxilary
        problem.add_equation("psiz - dz(psi)=0")  # auxilary
        problem.add_equation("psix - dx(psi)=0")  # auxilary
        problem.add_equation("zeta - dz(u) - dx(dx(psi))=0")  # zeta = grad^2(psi)

        problem.add_equation(" dt(v) - (dx(dx(v))*nu_h + dz(vz)*nu) - r*(1/H)*integ(v,'z')  +f*u =Jac_psi_v")
        problem.add_equation(" dt(u) - (dx(dx(zeta))*nu_h + zetazz*nu) - f*vz = Jac_psi_zeta ")


        # for 0th order
        if self.n == 0:
            problem.add_bc("vz(z='right') = (A/nu)*cos(x*k+ 3.14159/2)")  # wind forcing on 0th order boundary condition
        else:
            problem.add_bc("vz(z='right') = 0")  # no wind forcing for higher orders


        # Boundary conditions:
        problem.add_bc("psi(z='left') = 0")
        problem.add_bc("psi(z='right') = 0")
        problem.add_bc("vz(z='left') = 0")
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




    def analysis(self, FolderName):

        '''
        Obtains data values for state variable we want to analyze, stores the data as h5py file and performs
        specified tasks to the object. Saves variable data into initialized arrays for specified order.
        :rtype: object
        '''
        global psi_arr
        global t_arr
        global v_arr
        global dt
        global Jac_psi_zeta
        global Jac_psi_v

        folder_n = run_folder + 'snapshots_' + str(self.n) + '_n'
        folder_n_sub = 'snapshots_' + str(self.n) + '_n'

        # Analysis
        # snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=0.25, max_writes=50, mode=fh_mode)
        snapshots = solver.evaluator.add_file_handler(folder_n, iter=10, max_writes=150)
        snapshots.add_system(solver.state)
        snapshots.add_task("psi", layout='g', name='<psi>')
        snapshots.add_task("v", layout='g', name='<v>')  # saving variablesy\
        #solver.evaluator.evaluate_handlers([snapshots], world_time=0, wall_time=0, sim_time=solver.sim_time, timestep=dt, iteration = solver.iteration)

        # CFL
        CFL = flow_tools.CFL(solver, initial_dt=dt, cadence=10, safety=1,
                             max_change=1.5, min_change=0.5, max_dt=dt + 10, min_dt=dt - 10, threshold=0.05)
        CFL.add_velocities(('u', 'v'))

        # Flow properties
        flow = flow_tools.GlobalFlowProperty(solver, cadence=10)
        flow.add_property("sqrt(u*u)", name='U')

        t_count = 0
        # Main loop
        try:
            logger.info('Starting loop')
            while solver.proceed:
                dt = CFL.compute_dt()
                dt = solver.step(dt)

                if (solver.iteration - 1) % 10 == 0:
                    print('Iteration: %i, Time: %e, dt: %e' % (solver.iteration, solver.sim_time, dt))
                    logger.info('Iteration: %i, Time: %e, dt: %e' % (solver.iteration, solver.sim_time, dt))
                    logger.info(' U = %f' % flow.max('U'))
                    Jac_psi_zeta = self.Jn(t_count, self.n)[0]
                    Jac_psi_v = self.Jn(t_count, self.n)[1]
                    t_count += 1
        except:
            logger.error('Exception raised, triggering end of main loop.')
            raise

        n=1
        #folder = "snapshots"
        #with h5py.File(folder + '/' + folder + '_s' + str(n) + '.h5',
        #with h5py.File(folder + '/' + folder + '_s' + str(n) + '/' + folder+ '_s' + str(n) + '_p0.h5',
        #mode='r') as file:  # reading file

        with h5py.File(folder_n + '/' + folder_n_sub + '_s1/' + folder_n_sub + '_s1_p0.h5',
                       mode='r') as file:  # reading file
            psi = file['tasks']['<psi>']
            psi_arr[self.n,:, :, :] = np.array(file['tasks']['<psi>'])  # psi
            v_arr[self.n,:, :, :] = np.array(file['tasks']['<v>'])  # psi
            t_arr[:] = psi.dims[0]['sim_time']  # time array

    def plotting(self, folder, FileName):
        global max_vals
        '''
        Plots the corrected state variable at each order on x-z plane and saves to png
        :param FileName:
        :return:
        '''
        folder0 = 'out_0_n'
        x = np.linspace(0, L, nx)
        z = np.linspace(0, H, nz)
        X, Z = np.meshgrid(z, x)
        time_yrs = lambda t: round(t / 3.154e7, 2)
        time_mths = lambda t: round(t / 2.628e6, 2)
        # state.require_grid_space()
        # plot_tools.plot_bot_2d(state)
        # plt.savefig(FileName)

        fig = plt.figure(figsize=(10, 6))
        txt = "(H,L) = (" + "{:.1e}".format(H) + ", " + "{:.1e}".format(L) + ")"

        if self.n == 0:

            psi_arr_corrected[self.n, :, :, :] = psi_arr[0, :, :, :]
            # print(psi_arr[0,:,:])
            CS1 = plt.contourf(Z, X, psi_arr_corrected[0, n_time-1, :, :], 25, cmap='seismic')
            cbar = fig.colorbar(CS1)
        else:
            psi_arr_corrected[self.n, :, :, :] = psi_arr_corrected[self.n - 1, n_time-1, :, :] + psi_arr[self.n, :, :, :]
            # print(psi_arr_corrected[1,:,:])

            CS2 = plt.contourf(Z, X, psi_arr_corrected[self.n,n_time-1, :, :], 25, cmap='seismic')
            cbar = fig.colorbar(CS2)

        cbar.ax.set_ylabel('Streamfunction (kg/ms)')
        plt.ylabel('vertical depth')
        plt.xlabel('periodic x-axis (0,2$\pi$)')
        plt.title( '$\Psi$ Order ' + str(self.n) +' at t = ' + str(time_mths(t_arr[9])) + ' months')
        plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=12)
        plt.savefig(FileName)
        plt.close(fig)

        max_vals[self.n] = np.amax(psi_arr[self.n, n_time-1, :, :])


"""
_______________________________________________MAIN LOOP______________________________________________
"""


def main(n, R_e, R_g):
    figname = 'psi_o' + str(n) + '_n'

    if n == 0:  # Run to solve for N = 0:
        #  nu,f,r,tau = Rossby(R_e,R_g)
        s = Solver_n(Rossby(R_e, R_g)[0], Rossby(R_e, R_g)[1], Rossby(R_e, R_g)[2], Rossby(R_e, R_g)[3], 0)
        s.eqns('psi')
        s.analysis(run_folder)
        s.plotting(run_folder, run_folder + figname)

    else:

        s = Solver_n(Rossby(R_e, R_g)[0], Rossby(R_e, R_g)[1], Rossby(R_e, R_g)[2], Rossby(R_e, R_g)[3], n)
        s.eqns('psi')
        s.analysis(run_folder)
        s.plotting(run_folder, run_folder + figname)


###############_______________________________________

#    Main Loop
if __name__ == "__main__":
    # ekman rossby values run s.t. first one is the linearly converging state
    ek_rossby_vals = np.array([1.6]) # EKMAN ROSSBY NUMBER
    R_g = 0.1 # GEOSTROPHIC ROSSBY NUMBER
    plt.figure(figsize=(10, 6))
    start_time = time.time()
    count = 0

    p = tqdm(total=N, disable=False)
    divergence = None

    for j in ek_rossby_vals:  # runs for R_e in ek_rossby_vals
        count += 1
        max_vals = np.zeros(N + 1)  # reinitialize max_vals for each rossby number run

        for i in tqdm_notebook(range(0, N + 1)):  # runs for order i

            p.update(1)
            main(i, j, R_g)

            if i > 2:
                if max_vals[i] > max_vals[1]:
                    divergence = True
                    break
                else:
                    divergence = False

        p.close()

        if divergence == True:
            print("---------------------------------------------------------------------")
            print("Divergent for R_E = " + str(j) + " and R_G = " + str(R_g) + " at order n = " + str(i))
            print("---------------------------------------------------------------------")
            print("-------------------- PARAMETERS USED -------------------------")
            print("nu = ", Rossby(j, R_g)[0])
            print("f = ", Rossby(j, R_g)[1])
            print("r = ", Rossby(j, R_g)[2])
            print("tau = ", Rossby(j, R_g)[3])
            print("L = ", L)
            print("H = ", H)
            print("h_e = ", h_e)
            print("nx = ", nx)
            plt.plot(np.arange(0, i + 1), max_vals[0:i + 1] / max_vals[0],
                     label='$R_E = $' + str(j))  # normalized max_vals
            break
        else:
            print("----------- Rossby Run number " + str(count) + "/" + str(len(ek_rossby_vals)) + " DONE -----------")
            plt.plot(np.arange(0, N + 1), max_vals / max_vals[0], label='$R_E = $' + str(j))  # normalized max_vals

    if divergence == False:
        print("---------------------------------------------------------------------")
        print("---------------------------------------------------------------------")
        print("-------------------- PARAMETERS USED -------------------------")
        print("nu = ", Rossby(j, R_g)[0])
        print("f = ", Rossby(j, R_g)[1])
        print("r = ", Rossby(j, R_g)[2])
        print("tau = ", Rossby(j, R_g)[3])
        print("L = ", L)
        print("H = ", H)
        print("h_e = ", h_e)
        print("nx = ", nx)

    plt.grid(color='green', linestyle='--', linewidth=0.5)
    plt.ylabel('Normalized Max($\psi_{correction}$)')
    plt.xlabel('order of correction ')
    plt.title('Change in order of magnitude of $\psi_{correction}$ for $R_G$ = ' + str(
        R_g) + ')')
    plt.legend()
    plt.savefig(run_folder + 'psimax_order')

    print("Total Runtime: --- %s seconds ---" % (time.time() - start_time))

    R_e = ek_rossby_vals[0]
    lines = ["PARAMETERS FOR MODEL RUN", "(R_e, R_g) = (" + str(R_e) + ', ' + str(R_g) + ')',
             "nu = " + str(Rossby(R_e, R_g)[0]), "f = " + str(Rossby(R_e, R_g)[1]), "r = " +
             str(Rossby(R_e, R_g)[2]), "tau = " + str(Rossby(R_e, R_g)[3]),
             "L = " + str(L), "H = " + str(H), "h_e = " + str(h_e), "nx = " + str(nx), "nz = " + str(nz),
             "N = " + str(N)]

    with open(run_folder + 'read_me.txt', 'w') as f:
        for line in lines:
            f.write(line)
            f.write('\n')