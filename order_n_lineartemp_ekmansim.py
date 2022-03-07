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

#sys.path[0] = ""
from dedalus import public as de
from dedalus.extras import plot_tools

de.logging_setup.rootlogger.setLevel('ERROR')  # suppress logging msgs

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
print (L)
k = (2 * np.pi) / (L)
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



x_basis = de.Fourier('x', nx, interval=(0, L))
z_basis = de.Chebyshev('z', nz, interval=(0, H))
domain = de.Domain([x_basis, z_basis], grid_dtype=np.float64)

# declaring domain and variables for the problem
problem = de.LBVP(domain, variables=['psi', 'u', 'v', 'vx',
                                     'vz', 'zeta',
                                     'zetaz', 'zetax', 'psix', 'psiz'])

# Initializing arrays to store values at each order (adding one to the order column to account for indexing limits)
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
max_vals = np.zeros(N + 1)


class Solver_n:
    def __init__(self, visc, coriolis, damp, wind, order):

        '''
        Class
        :param visc:
        :param coriolis:
        :param damp:
        :param wind:
        :param jacobian: J(psi,zeta) -- only need this parameter for N=1 since Jn(self) calculates this for N>1
        :param jacobian2: J(psi,v) -- only need this parameter for N=1 since Jn(self) calculates this for N>1
        :param order:
        '''

        global nu
        global r
        global tau
        # global psi_arr
        # global psi_x_arr
        # global psi_z_arr
        # global zeta_arr
        # global zeta_x_arr
        # global zeta_z_arr
        # global psi_arr_corrected
        # global v_arr
        # global v_x_arr
        # global v_z_arr

        self.nu = visc
        self.f = coriolis
        self.r = damp  # damping parameter to dampen as depth increases
        self.tau = wind
        self.n = order


        nu = self.nu
        r = self.r
        tau = self.tau

    def Jn(self, n_solve):

        '''
        Calculates the nonlinear terms of order n

        :param self:
        :param n: order to solve J to
        :return: [J(psi,zeta), J(psi,v)]
        '''
        J1 = 0
        J2 = 0
        folder0 = 'out_0_n'
        global psi_arr

        if n_solve==0:
            return J1,J2
        # elif n==1:
        #     # loading Jacobians (nonlinear terms) from the 0th order analysis tasks
        #     with h5py.File(folder0 + '/' + folder0 + '_s1/' + folder0 + '_s1_p0.h5', mode='r') as file:  # reading file
        #         J0_1 = np.array(file['tasks']['<J>'])
        #         J0_2 = np.array(file['tasks']['<J_psi_v>'])
        #
        #
        #     # initializing and saving loaded jacobians into dedalus-readable fields
        #     Jac = domain.new_field()
        #     gslices = domain.dist.grid_layout.slices(scales=1)
        #     Jac['g'] = J0_1[0, :, :][gslices[0]]
        #
        #     Jac_psi_v = domain.new_field()
        #     gslices = domain.dist.grid_layout.slices(scales=1)
        #     Jac_psi_v['g'] = J0_2[0, :, :][gslices[0]]
        #
        #     return Jac,Jac_psi_v

        else:
            n=n_solve-1
            #print ("n_solve-1=",n)
            for j in range(0, n+1):
                J1 += psi_x_arr[j, :] * zeta_z_arr[n - j, :] - psi_z_arr[j, :] * zeta_x_arr[n - j, :]
                J2 += psi_x_arr[j, :] * v_z_arr[n - j, :] - psi_z_arr[j, :] * v_x_arr[n - j, :]


            Jac_temp1 = domain.new_field()
            gslices = domain.dist.grid_layout.slices(scales=1)
            Jac_temp1['g'] = J1[gslices[0]]

            Jac_temp2 = domain.new_field()
            gslices = domain.dist.grid_layout.slices(scales=1)
            Jac_temp2['g'] = J2[gslices[0]]

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

        #print("n_solve=",self.n)

        problem = de.LBVP(domain, variables=['psi', 'u', 'v', 'vx',
                                             'vz', 'zeta',
                                             'zetaz', 'zetax', 'psix', 'psiz'])

        a_visc = ((nx/2)**2)/(h_e**2)
        # setting up all parameters
        problem.parameters['nu'] = self.nu  # viscosity
        problem.parameters['nu_h'] = a_visc * self.nu
        problem.parameters['f'] = self.f  # coriolis param
        problem.parameters['r'] = self.r  # damping param
        problem.parameters['A'] = self.tau  # forcing param
        problem.parameters['H'] = H
        problem.parameters['k'] = k


        problem.parameters['Jac'] = self.Jn(self.n)[0]  # self.j = self.j2 = 0 for 0th order solution
        problem.parameters['Jac_psi_v'] = self.Jn(self.n)[1]

        # axuliary equations:
        problem.add_equation("u - psiz = 0")
        problem.add_equation("vz - dz(v) = 0")  # auxilary
        problem.add_equation("vx - dx(v) = 0")  # auxilary
        problem.add_equation("zetaz - dz(zeta)=0")  # auxilary
        problem.add_equation("zetax - dx(zeta)=0")  # auxilary
        problem.add_equation("psiz - dz(psi)=0")  # auxilary
        problem.add_equation("psix - dx(psi)=0")  # auxilary
        problem.add_equation("zeta - dz(u) - dx(dx(psi))=0")  # zeta = grad^2(psi)

        problem.add_equation("(dx(dx(v))*nu_h + dz(vz)*nu) - r*(1/H)*integ(v,'z')  -f*u = Jac_psi_v")  # nu* grad^2 v  - fu=0
        problem.add_equation("(dx(dx(zeta))*nu_h + dz(zetaz)*nu) + f*vz = Jac")  # nu* grad^2 zeta + fv_z=0

        # for 0th order
        if self.n == 0:
            problem.add_bc("vz(z='right') = (A/nu)*sin(x*k)")  # wind forcing on 0th order boundary condition
        else:
            problem.add_bc("vz(z='right') = 0")  # no wind forcing for higher orders
            #print('bc')
        # Boundary conditions:
        problem.add_bc("psi(z='left') = 0")
        problem.add_bc("psi(z='right') = 0")
        problem.add_bc("vz(z='left') = 0")
        problem.add_bc("dz(u)(z='left') = 0")
        problem.add_bc("dz(u)(z='right') = 0")

        # Building Solver:
        solver = problem.build_solver()
        solver.solve()
        state = solver.state[state_var]

    def analysis(self, FolderName):

        '''
        Obtains data values for state variable we want to analyze, stores the data as h5py file and performs
        specified tasks to the object. Saves variable data into initialized arrays for specified order.
        :rtype: object
        '''
        global psi_arr
        global psi_x_arr
        global psi_z_arr
        global zeta_arr
        global zeta_x_arr
        global zeta_z_arr
        global v_x_arr
        global v_z_arr
        global v_arr

        folder_n = 'out_' + str(self.n) + '_n'


        out = solver.evaluator.add_file_handler(folder_n)  # storing output into file with specified name
        out.add_system(solver.state)


        out.add_task("psi", layout='g', name='<psi>')  # saving variables
        out.add_task("psix", layout='g', name='<psix>')  # saving variables
        out.add_task("psiz", layout='g', name='<psiz>')  # saving variables
        out.add_task("zetax", layout='g', name='<zetax>')  # saving variables
        out.add_task("zetaz", layout='g', name='<zetaz>')  # saving variables
        out.add_task("zeta", layout='g', name='<zeta>')  # saving variables
        out.add_task("u", layout='g', name='<u>')
        out.add_task("vx", layout='g', name='<vx>')
        out.add_task("vz", layout='g', name='<vz>')
        out.add_task("v", layout='g', name='<v>')
        out.add_task("psix* zetaz - psiz * zetax", layout='g', name='<J>')
        out.add_task("u*vx  + psix*vz", layout='g', name='<J_psi_v>')
        # evaluates the tasks declared above:
        solver.evaluator.evaluate_handlers([out], world_time=0, wall_time=0, sim_time=0, timestep=0, iteration=0)

        with h5py.File(folder_n + '/' + folder_n + '_s1/' + folder_n + '_s1_p0.h5',
                       mode='r') as file:  # reading file

            psi_arr[self.n, :, :] = np.array(file['tasks']['<psi>'])  # psi
            zeta_arr[self.n, :, :] = np.array(file['tasks']['<zeta>'])  # zeta
            psi_x_arr[self.n, :, :] = np.array(file['tasks']['<psix>'])  # d/dx (psi)
            psi_z_arr[self.n, :, :] = np.array(file['tasks']['<psiz>'])  # d/dz (psi)
            zeta_arr[self.n, :, :] = np.array(file['tasks']['<zeta>'])  # zeta
            zeta_x_arr[self.n, :, :] = np.array(file['tasks']['<zetax>'])  # d/dx (zeta)
            zeta_z_arr[self.n, :, :] = np.array(file['tasks']['<zetaz>'])  # d/dz (zeta)
            v_x_arr[self.n, :, :] = np.array(file['tasks']['<vx>'])  #
            v_z_arr[self.n, :, :] = np.array(file['tasks']['<vz>'])  #
            v_arr[self.n, :, :] = np.array(file['tasks']['<v>'])  #


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

        # state.require_grid_space()
        # plot_tools.plot_bot_2d(state)
        # plt.savefig(FileName)

        fig = plt.figure(figsize=(10, 6))
        txt = "(H,L) = (" + "{:.1e}".format(H) + ", " + "{:.1e}".format(L) + ")"

        if self.n == 0:

            psi_arr_corrected[self.n, :, :] = psi_arr[0, :, :]
            # print(psi_arr[0,:,:])
            CS1 = plt.contourf(Z, X, psi_arr_corrected[0, :, :], cmap='seismic')
            cbar = fig.colorbar(CS1)
        else:
            psi_arr_corrected[self.n, :, :] = psi_arr_corrected[self.n - 1, :, :] + psi_arr[self.n, :, :]
            # print(psi_arr_corrected[1,:,:])

            CS2 = plt.contourf(Z, X, psi_arr_corrected[self.n, :, :], cmap='seismic')
            cbar = fig.colorbar(CS2)

        cbar.ax.set_ylabel('Streamfunction (kg/ms)')
        plt.ylabel('vertical depth')
        plt.xlabel('periodic x-axis (0,2$\pi$)')
        plt.title(
            'Streamfunction Order ' + str(self.n) + ' with $\\nu$ =' + "{:.1e}".format(nu) + ' , r= ' + "{:.1e}".format(
                r) + ' , $\\tau$=' + "{:.1e}".format(tau))
        plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=12)
        plt.savefig(FileName)
        plt.close(fig)

        max_vals[self.n] = np.amax(psi_arr[self.n, :, :])







"""
_______________________________________________MAIN LOOP______________________________________________
"""


def main(n,R_e,R_g):
    folder0 =  'out_0_n'

    folder = 'out_' + str(n) + '_n'
    figname = 'psi_o' + str(n) + '_n'

    if n==0: # Run to solve for N = 0:
        #  nu,f,r,tau = Rossby(R_e,R_g)
        s = Solver_n(Rossby(R_e,R_g)[0],Rossby(R_e,R_g)[1],Rossby(R_e,R_g)[2],Rossby(R_e,R_g)[3], 0)
        s.eqns('psi')
        s.analysis(folder)
        s.plotting(folder, figname)

    else:

        s = Solver_n(Rossby(R_e,R_g)[0],Rossby(R_e,R_g)[1],Rossby(R_e,R_g)[2],Rossby(R_e,R_g)[3], n)
        s.eqns('psi')
        s.analysis(folder)
        s.plotting(folder, figname)



###############_______________________________________

#    Main Loop
if __name__ == "__main__":
    #ekman rossby values run s.t. first one is the linearly converging state
    ek_rossby_vals = np.array([0.9])
    R_g = 0.1
    plt.figure(figsize=(10, 6))
    start_time = time.time()
    count=0

    p = tqdm(total=N, disable=False)
    divergence = None

    for j in ek_rossby_vals:  #runs for R_e in ek_rossby_vals
        count += 1
        max_vals = np.zeros(N + 1) #reinitialize max_vals for each rossby number run

        for i in tqdm_notebook(range(0,N + 1)):  #runs for order i

            p.update(1)
            main(i,j,R_g)

            if i>2:
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
            print("nu = " , Rossby(j,R_g)[0])
            print("f = " , Rossby(j,R_g)[1])
            print("r = " , Rossby(j,R_g)[2])
            print("tau = " , Rossby(j,R_g)[3])
            print ("L = ", L)
            print ("H = ", H)
            print("h_e = ", h_e)
            print("nx = ", nx)
            plt.plot(np.arange(0, i + 1), max_vals[0:i+1] / max_vals[0], label='$R_E = $' + str(j))  # normalized max_vals
            break
        else:
            print("----------- Rossby Run number " + str(count) + "/" + str(len(ek_rossby_vals)) + " DONE -----------" )
            plt.plot(np.arange(0, N + 1), max_vals / max_vals[0], label= '$R_E = $' + str(j))  # normalized max_vals

    if divergence == False:
        print("---------------------------------------------------------------------")
        print("Divergent for R_E = " + str(j) + " and R_G = " + str(R_g) + " at order n = " + str(i))
        print("---------------------------------------------------------------------")
        print("-------------------- PARAMETERS USED -------------------------")
        print("nu = " , Rossby(j,R_g)[0])
        print("f = " , Rossby(j,R_g)[1])
        print("r = " , Rossby(j,R_g)[2])
        print("tau = " , Rossby(j,R_g)[3])
        print ("L = ", L)
        print ("H = ", H)
        print("h_e = ", h_e)
        print("nx = ", nx)
        
        
    plt.grid(color='green', linestyle='--', linewidth=0.5)
    plt.ylabel('Normalized Max($\psi_{correction}$)')
    plt.xlabel('order of correction ')
    plt.title('Change in order of magnitude of $\psi_{correction}$ for $R_G$ = ' + str(
            R_g) + ')')
    plt.legend()
    plt.savefig('psimax_order')

    print("Total Runtime: --- %s seconds ---" % (time.time() - start_time))
