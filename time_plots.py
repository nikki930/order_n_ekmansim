import numpy as np
import time
import h5py
import matplotlib.pyplot as plt
import dedalus.public as d3
import matplotlib
import ffmpeg
# import cv2
import os

# Parameters
nx = 512 # fourier resolution
nz = 128  # chebyshev resolution

H = 10  # depth of water in meters
h_e = 1 #ekman thickness
dh = h_e/H  # dh = ekman thickness divided by H
da = 0.01  # aspect ratio = ratio of height to length
f = 1e-4  # coriolis param in 1/s

L_func = lambda H, delta_a: H / delta_a
L = L_func(H, da)
k = (2 * np.pi) / (L)

Re = 1.6
Rg=0.1

sim_time = 50
n_snaps = 50
dt = 0.125
psi_arr = np.zeros((n_snaps,nx, nz))

zcoord = d3.Coordinate('z')
dist = d3.Distributor(zcoord)
z_basis = d3.Chebyshev(zcoord, size=nz, bounds=(0, H))

x = np.linspace(0, L, nx)
X, Z = np.meshgrid(dist.local_grid(z_basis),x )

s_num = 1
for n in range(1,s_num+1):

    folder = "snapshots"
    with h5py.File(folder + '/' + folder + '_s' + str(n) + '.h5',
                   mode='r') as file:  # reading file

        psi_arr[:, :, :] = np.array(file['tasks']['<psi>'])  # psi



    for i in range(n_snaps):

        fig,ax= plt.subplots(constrained_layout=True)
        CM= plt.pcolormesh(Z, X, psi_arr[i,:,:], shading='gouraud',cmap='PRGn')
        cbar = fig.colorbar(CM)
        plt.ylabel('vertical depth')
        plt.xlabel('periodic x-axis (0,2$\pi$)')
        plt.title('$\psi(t=0)$')
        plt.savefig("images_ivp/s" + str(n) + "_" + str(i) + '.png')
        plt.close(fig)



# (
#     ffmpeg
#         .input('/images_ivp/s1_*.png', pattern_type='glob', framerate=25, analyzeduration = 100, probesize =100)
#         .output('movie.mp4')
#         .run()
# )



# Animate IVP
from matplotlib.animation import FuncAnimation, FFMpegWriter
from itertools import count

fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
plt.subplot()
plt.pcolormesh(Z, X, psi_arr[i,:,:], shading='gouraud',cmap='PRGn')
plt.colorbar()

i = 0

def plot_frame(frame):
    global i
    im =plt.pcolormesh(Z, X, psi_arr[i,:,:], shading='gouraud',cmap='PRGn')
    print(i)
    i += 1
    return im,


ani = FuncAnimation(fig, plot_frame, interval=17, blit="True", save_count=20, repeat=False)
ani.save('s1_video.mov', writer="ffmpeg")
plt.show()



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
