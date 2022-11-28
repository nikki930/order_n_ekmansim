import numpy as np
import time
import h5py
import matplotlib.pyplot as plt
import dedalus.public as d3
import matplotlib
import imageio
import os

# Parameters
nx = 512 # fourier resolution
nz = 68 # chebyshev resolution

H = 200  # depth of water in meters
h_e = 20 #ekman thickness
dh = h_e/H  # dh = ekman thickness divided by H
da = 0.01  # aspect ratio = ratio of height to length
f = 1e-4    # coriolis param in 1/s

L_func = lambda H, delta_a: H / delta_a
L = L_func(H, da)
k = (2 * np.pi) / (L)

Re = 1.75
Rg=0.1

n_snaps = 582

psi_arr = np.zeros((n_snaps,nx, nz))
v_arr = np.zeros((n_snaps, nx, nz))
t_arr = np.zeros(n_snaps)

#zcoord = d3.Coordinate('z')
#dist = d3.Distributor(zcoord)
#z_basis = d3.Chebyshev(zcoord, size=nz, bounds=(0, H))

x = np.linspace(0, L, nx)
z = np.linspace(0, H, nz)
X, Z = np.meshgrid(z,x )

# fig, ax = plt.subplots(constrained_layout=True)
# plt.plot(x, v_arr[10, :, 68])
# plt.ylabel('vertical depth')
# plt.xlabel('periodic x-axis (0,2$\pi$)')
# plt.title('$v(t=0)$')
# plt.show()
# plt.close(fig)
run_folder = 'Re_big/ivp/'
time_yrs = lambda t: round(t/3.154e7,2)
time_mths = lambda t: round(t/2.628e6,4)
time_days = lambda t: round(t/86400,2)


run_folder = 'Re_big/ivp/snapshots'
folder_n = run_folder
folder_n_sub = 'snapshots'
folder = 'Re_big/ivp/'

with h5py.File(folder_n + '/' + folder_n_sub + '_s1.h5',mode = 'r') as file:  #when using more than one core
#with h5py.File(folder_n + '/' + folder_n_sub + '_s1/' + folder_n_sub + '_s1_p0.h5', mode = 'r') as file:

    psi = file['tasks']['<psi>']
    psi_arr[:,:, :] = np.array(file['tasks']['<psi>'])  # psi
    v_arr[:,:, :] = np.array(file['tasks']['<v>'])  # psi
    t_arr[:] = psi.dims[0]['sim_time'] #time array

plt_idx = np.arange(1,n_snaps,10)
print(plt_idx)
def plotting():
    for t in plt_idx:
        maxval_psi=6
        maxval_v = 0.65
        #
        fig,ax= plt.subplots(constrained_layout=True)
        #CM= plt.pcolormesh(Z, X, psi_arr[i,:,:], shading='gouraud',cmap='PRGn', vmin=-maxval_psi, vmax=maxval_psi)
        CS = plt.contour(Z, X, psi_arr[t, :, :], 30, colors='k')
        CM = plt.pcolormesh(Z, X, psi_arr[ t, :, :], shading='gouraud', cmap='PRGn', vmin=-maxval_psi, vmax=maxval_psi)
        cbar = fig.colorbar(CM)
        cbar.ax.set_ylabel('Streamfunction')
        plt.ylabel('vertical depth')
        plt.xlabel('periodic x-axis (0,2$\pi$)')
        plt.title('$\psi$ at t = ' + str(time_days(t_arr[t])) + ' days')
        plt.savefig("Re_big/ivp/psi/psi_t" + str(t) + '.png')
        plt.close(fig)
    #
        fig, ax = plt.subplots(constrained_layout=True)
        #CM = plt.pcolormesh(Z, X, v_arr[i, :, :], shading='gouraud', cmap='PRGn', vmin=-maxval_v, vmax=maxval_v)
        CS = plt.contour(Z, X, v_arr[t, :, :], 30, colors='k')
        CM = plt.pcolormesh(Z, X, v_arr[t, :, :], shading='gouraud', cmap='PRGn', vmin=-maxval_psi, vmax=maxval_psi)
        cbar = fig.colorbar(CM)
        cbar.ax.set_ylabel('Velocity')
        plt.ylabel('vertical depth')
        plt.xlabel('periodic x-axis (0,2$\pi$)')
        plt.title('$v$ at t = ' + str(time_days(t_arr[t])) + ' days')
        plt.savefig("Re_big/ivp/v/v_t" + str(t) + '.png')
        plt.close(fig)
plotting()


#____________________________________________other animation methods_________________________________________________
def animate(variable):

    path = 'Re_big/ivp/' + str(variable) + '/'
    image_folder = os.fsencode(path)
    video_name = 'psi_noise.mov'

    images = []
    filenames = []

    for file in os.listdir(image_folder):
        filename = os.fsdecode(file)
        if filename.endswith( ('.jpeg', '.png', '.gif') ):
            filenames.append(os.path.join(path, filename))
    filenames.sort(key=lambda f: int(''.join(filter(str.isdigit, f)))) #sort the frames in order

    images = list(map(lambda filename: imageio.imread(filename), filenames))

    imageio.mimsave(os.path.join('Re_big/ivp/' +str(variable) + '/' + video_name), images) # modify duration as needed

animate('psi')

#____________________________________________other animation methods_________________________________________________
# # Animate IVP
# from matplotlib.animation import FuncAnimation, FFMpegWriter
# from itertools import count
#
# fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
# # plt.subplot()
# # plt.pcolormesh(Z, X, psi_arr[i,:,:], shading='gouraud',cmap='PRGn')
# # plt.colorbar()
#
# i = 0
#
# def plot_frame(frame):
#     global i
#     im =plt.pcolormesh(Z, X, v_arr[i,:,:], shading='gouraud',cmap='PRGn')
#     print(i)
#     i += 1
#     return im,
#
#
# ani = FuncAnimation(fig, plot_frame, interval=17, blit="True", save_count=20, repeat=False)
# ani.save('s1_video.mov', writer="ffmpeg")
# plt.show()