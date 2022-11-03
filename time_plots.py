import numpy as np
import time
import h5py
import matplotlib.pyplot as plt
import dedalus.public as d3
import matplotlib
import ffmpeg
import os

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


n_snaps = 10

psi_arr = np.zeros((n_snaps,nx, nz))
v_arr = np.zeros((n_snaps, nx, nz))
t_arr = np.zeros(n_snaps)

zcoord = d3.Coordinate('z')
dist = d3.Distributor(zcoord)
z_basis = d3.Chebyshev(zcoord, size=nz, bounds=(0, H))

x = np.linspace(0, L, nx)
X, Z = np.meshgrid(dist.local_grid(z_basis),x )

# fig, ax = plt.subplots(constrained_layout=True)
# plt.plot(x, v_arr[10, :, 68])
# plt.ylabel('vertical depth')
# plt.xlabel('periodic x-axis (0,2$\pi$)')
# plt.title('$v(t=0)$')
# plt.show()
# plt.close(fig)

time_yrs = lambda t: round(t/3.154e7,2)
time_mths = lambda t: round(t/2.628e6,2)
s_num = 1
for n in range(1,s_num+1):

    folder = "Re_big/ivp/linear/"
    sub_folder = "snapshots"
    #with h5py.File(folder + '/' + folder + '_s' + str(n) + '.h5',
    with h5py.File(folder + sub_folder + '/' + sub_folder + '_s' + str(n) + '/' + sub_folder+ '_s' + str(n) + '_p0.h5',
                   mode='r') as file:  # reading file
        psi = file['tasks']['<psi>']
        psi_arr[:, :, :] = np.array(file['tasks']['<psi>'])  # psi
        v_arr[:, :, :] = np.array(file['tasks']['<v>'])  # psi
        t_arr[:] = psi.dims[0]['sim_time'] #time array


    #maxval_psi=0.0006
    maxval_v = 1e-5
    for i in range(n_snaps):
    #
        fig,ax= plt.subplots(constrained_layout=True)
        #CM= plt.pcolormesh(Z, X, psi_arr[i,:,:], shading='gouraud',cmap='PRGn', vmin=-maxval_psi, vmax=maxval_psi)
        CM = plt.pcolormesh(Z, X, psi_arr[i, :, :], shading='gouraud', cmap='PRGn')
        cbar = fig.colorbar(CM)
        plt.ylabel('vertical depth')
        plt.xlabel('periodic x-axis (0,2$\pi$)')
        plt.title('$\psi$ at t = ' + str(time_mths(t_arr[i])) + ' months')
        plt.savefig(folder + "psi_test_s" + str(n) + "_" + str(i) + '.png')
        plt.close(fig)
    #
        fig, ax = plt.subplots(constrained_layout=True)
        #CM = plt.pcolormesh(Z, X, v_arr[i, :, :], shading='gouraud', cmap='PRGn', vmin=-maxval_v, vmax=maxval_v)
        CM = plt.pcolormesh(Z, X, v_arr[i, :, :], shading='gouraud', cmap='PRGn')
        cbar = fig.colorbar(CM)
        plt.ylabel('vertical depth')
        plt.xlabel('periodic x-axis (0,2$\pi$)')
        plt.title('$v$ at t = ' + str(time_mths(t_arr[i])) + ' months')
        plt.savefig(folder + "v_test_s" + str(n) + "_" + str(i) + '.png')
        plt.close(fig)



# # Animate IVP
# from matplotlib.animation import FuncAnimation, FFMpegWriter
# from itertools import count
#
# fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
# plt.subplot()
# plt.pcolormesh(Z, X, psi_arr[i,:,:], shading='gouraud',cmap='PRGn')
# plt.colorbar()
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


#____________________________________________other animation methods_________________________________________________

# image_folder = 'images_ivp'
# video_name = 's1_video.avi'
#
# images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
# frame = cv2.imread(os.path.join(image_folder, images[0]))
# height, width, layers = frame.shape
#
# video = cv2.VideoWriter(video_name, 0, 1, (width,height))
#
# for image in images:
#     video.write(cv2.imread(os.path.join(image_folder, image)))
#
# cv2.destroyAllWindows()
# video.release()

# (
#     ffmpeg
#         .input('/images_ivp/s1_*.png', pattern_type='glob', framerate=25, analyzeduration = 100, probesize =100)
#         .output('movie.mp4')
#         .run()
# )
