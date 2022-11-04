import numpy as np
import time
import h5py
import matplotlib.pyplot as plt
import dedalus.public as d3
import matplotlib
#import ffmpeg
import os

# Parameters
nx = 512 # fourier resolution
nz = 128  # chebyshev resolution

H = 100  # depth of water in meters
h_e = 10 #ekman thickness
dh = h_e/H  # dh = ekman thickness divided by H
da = 0.01  # aspect ratio = ratio of height to length
f = 1e-4    # coriolis param in 1/s

L_func = lambda H, delta_a: H / delta_a
L = L_func(H, da)
k = (2 * np.pi) / (L)

Re = 1.6
Rg=0.1

N=10
n_snaps = 10

psi_arr = np.zeros((N,n_snaps,nx, nz))
psi_arr_corrected= np.zeros((N,n_snaps,nx, nz))
v_arr = np.zeros((N,n_snaps, nx, nz))
v_arr_corrected = np.zeros((N,n_snaps, nx, nz))
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
time_mths = lambda t: round(t/2.628e6,2)

for n in range (0,N-1):

    folder_n = run_folder + 'snapshots_' + str(n) + '_n'
    folder_n_sub = 'snapshots_' + str(n)  + '_n'

    #with h5py.File(folder + '/' + folder + '_s' + str(n) + '.h5',
    #with h5py.File(folder + '/' + folder + '_s' + str(n) + '/' + folder+ '_s' + str(n) + '_p0.h5',
     #              mode='r') as file:  # reading file
    with h5py.File(folder_n + '/' + folder_n_sub + '_s1/' + folder_n_sub + '_s1_p0.h5',
                   mode='r') as file:  # reading file
        psi = file['tasks']['<psi>']
        psi_arr[n,:,:, :] = np.array(file['tasks']['<psi>'])  # psi
        v_arr[n ,:,:, :] = np.array(file['tasks']['<v>'])  # psi
        t_arr[:] = psi.dims[0]['sim_time'] #time array
    if n == 0:
        psi_arr_corrected[n, :, :] = psi_arr[0, :, :]
        v_arr_corrected[n, :, :] = v_arr[0, :, :]

    else:
        psi_arr_corrected[n, :,:, :] = psi_arr_corrected[n - 1,:, :, :] + psi_arr[n, :,:, :]
        v_arr_corrected[n, :,:, :] = v_arr_corrected[n - 1, :,:, :] + v_arr[n, :,:, :]

print(psi_arr_corrected[5,3, :, :]-psi_arr_corrected[3,3, :, :])

n=5
for t in range(1, n_snaps):
    #maxval_psi=0.0006
    maxval_v = 1e-5
    #
    fig,ax= plt.subplots(constrained_layout=True)
    #CM= plt.pcolormesh(Z, X, psi_arr[i,:,:], shading='gouraud',cmap='PRGn', vmin=-maxval_psi, vmax=maxval_psi)
    CS = plt.contour(Z, X, psi_arr_corrected[n,t, :, :], 30, colors='k')
    CM = plt.pcolormesh(Z, X, psi_arr_corrected[n, t, :, :], shading='gouraud', cmap='PRGn')
    cbar = fig.colorbar(CM)
    cbar.ax.set_ylabel('Streamfunction')
    plt.ylabel('vertical depth')
    plt.xlabel('periodic x-axis (0,2$\pi$)')
    plt.title('$\psi$ at t = ' + str(time_mths(t_arr[t])) + ' months')
    plt.savefig("Re_big/ivp/psi_test_o" + str(n) + "_t" + str(t) + '.png')
    plt.close(fig)
#
    fig, ax = plt.subplots(constrained_layout=True)
    #CM = plt.pcolormesh(Z, X, v_arr[i, :, :], shading='gouraud', cmap='PRGn', vmin=-maxval_v, vmax=maxval_v)
    CS = plt.contour(Z, X, v_arr_corrected[n,t, :, :], 30, colors='k')
    CM = plt.pcolormesh(Z, X, v_arr_corrected[n, t, :, :], shading='gouraud', cmap='PRGn')
    cbar = fig.colorbar(CM)
    cbar.ax.set_ylabel('Velocity')
    plt.ylabel('vertical depth')
    plt.xlabel('periodic x-axis (0,2$\pi$)')
    plt.title('$v$ at t = ' + str(time_mths(t_arr[t])) + ' months')
    plt.savefig("Re_big/ivp/v_test_o" + str(n) + "_t" + str(t) + '.png')
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
