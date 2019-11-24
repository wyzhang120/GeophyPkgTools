import os
import segyio
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import AutoMinorLocator
from skimage.feature import canny
from skimage.transform import hough_line
from skimage.filters import sobel_v, gaussian


datadir = 'D:\Geophysics\Project\Marmousi\model'
fname = 'MODEL_P-WAVE_VELOCITY_1.25m.segy'
# nx = 13601, nz = 2801
dx = 1.25

def plot_marmousi(datadir, fname):
    with segyio.open(os.path.join(datadir, fname), mode='r') as f:
        vp = f.xline
        nx, nz = vp.shape
        xmax = dx * (nx - 1)
        zmax = dx * (nz - 1)
        ext = (0, xmax, zmax, 0)
        fig, ax = plt.subplots()
        fig.set_size_inches([12, 3])
        img = ax.imshow(vp[0].T/1000, extent=ext, cmap='jet', vmin=1, vmax=5.5)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cbar = fig.colorbar(img, cax=cax, spacing='uniform')
        cbar.set_label('km/s')
        ax.set_title('Mamousi Vp')
        ax.set_xticks(np.arange(0, xmax + 1000, 1000))
        ax.set_yticks(np.arange(0, zmax + 500, 500))
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.tick_params(axis='both', direction='out', length=7)
        ax.grid(True, which='major', axis='both', color='w', linestyle='--')
        plt.tight_layout(pad=0)
        fig.savefig(os.path.join(datadir, 'MamousiVp.pdf'))
        plt.show()
        plt.close(fig)


def window_marmousi(datadir, fname, figname, x0=8500, x1=11500, z0=1000, z1=3500, txtlist=None):
    with segyio.open(os.path.join(datadir, fname), mode='r') as f:
        vp = f.xline
        idx0 = int(np.round(x0 / dx))
        nx = int(np.round((x1 - x0) / dx) + 1)
        xmin = idx0 * dx
        xmax = xmin + (nx - 1) * dx
        idz0 = int(np.round(z0 / dx))
        nz = int(np.round(z1 - z0) / dx + 1)
        zmin = idz0 * dx
        zmax = zmin + (nz - 1) * dx
        ext = (xmin, xmax, zmax, zmin)
        mod = vp[0][idx0:idx0 + nx, idz0:idz0 + nz]
        fig, ax = plt.subplots()
        img = ax.imshow(mod.T / 1000, extent=ext, vmin=1, vmax=5.5)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cbar = fig.colorbar(img, cax=cax, spacing='uniform')
        ax.set_xticks(np.arange(xmin, xmax + 1, 500))
        ax.set_yticks(np.arange(zmin, zmax + 1, 500))
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.tick_params(axis='both', direction='out', length=7)
        ax.grid(True, which='major', axis='both', color='w', linestyle='--')
        ax.set_ylabel('Z [m]', labelpad=15)
        cbar.set_label('km/s')
        if txtlist is not None:
            for itxt in txtlist:
                x, z, txt = itxt
                ax.text(x, z, txt, horizontalalignment='center', verticalalignment='center')
        plt.tight_layout(pad=0)
        fig.savefig(os.path.join(datadir, figname + '.pdf'))
        plt.show()


def crop_sgy(datadir, fname, x0, x1, z0, z1):
    with segyio.open(os.path.join(datadir, fname), mode='r') as f:
        vp = f.xline
        idx0 = int(np.round(x0 / dx))
        nx = int(np.round((x1 - x0) / dx) + 1)
        xmin = idx0 * dx
        xmax = xmin + (nx - 1) * dx
        idz0 = int(np.round(z0 / dx))
        nz = int(np.round(z1 - z0) / dx + 1)
        zmin = idz0 * dx
        zmax = zmin + (nz - 1) * dx
        ext = (xmin, xmax, zmax, zmin)
        mod = vp[0][idx0:idx0 + nx, idz0:idz0 + nz].T
        return ext, mod


def get_dips(datadir, fname, x0=8500, x1=11500, z0=1000, z1=3500):
    with segyio.open(os.path.join(datadir, fname), mode='r') as f:
        # vp = f.xline
        # idx0 = int(np.round(x0 / dx))
        # nx = int(np.round((x1 - x0) / dx) + 1)
        # xmin = idx0 * dx
        # xmax = xmin + (nx - 1) * dx
        # idz0 = int(np.round(z0 / dx))
        # nz = int(np.round(z1 - z0) / dx + 1)
        # zmin = idz0 * dx
        # zmax = zmin + (nz - 1) * dx
        # ext = (xmin, xmax, zmax, zmin)
        # mod = vp[0][idx0:idx0 + nx, idz0:idz0 + nz].T
        ext, mod = crop_sgy(datadir, fname, x0, x1, z0, z1)
        edges = canny(mod, sigma=3, low_threshold=0.05, high_threshold=0.85, use_quantiles=True)
        # edges = sobel_v(mod)
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(mod / 1000, extent=ext, vmin=1., vmax=5.5)
        ax[1].imshow(edges, cmap='gray', extent=ext)
        plt.show()
        h, theta, d = hough_line(edges)
        idxh = np.array(np.unravel_index(np.argsort(np.ravel(h)), h.shape))
        idx = idxh[1][::-1]
        print(np.degrees(theta[idx[:30]]))


def blur_image(datadir, fname, sigma, repeat=1, x0=8500, x1=11500, z0=1000, z1=3500):
    ext, mod = crop_sgy(datadir, fname, x0, x1, z0, z1)
    mod_blur = np.copy(mod)
    for i in range(repeat):
        mod_blur = gaussian(mod_blur, sigma)
    fig, ax = plt.subplots(1, 2)
    img0 = ax[0].imshow(mod / 1000, extent=ext, vmin=1., vmax=5.5)
    divider0 = make_axes_locatable(ax[0])
    cax0 = divider0.append_axes('right', size='5%', pad=0.05)
    cbar0 = fig.colorbar(img0, cax=cax0, spacing='uniform')
    cbar0.set_label('km/s')
    img = ax[1].imshow(mod_blur / 1000, extent=ext, vmin=1., vmax=5.5)
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(img, cax=cax, spacing='uniform')
    cbar.set_label('km/s')
    plt.tight_layout(pad=0)
    plt.show()


# plot model
# plot_marmousi(datadir, fname)

# plot the windowed model
# window_marmousi(datadir, fname, figname='MamousiVp_Reservoirs',
#                 x0=8500, x1=11500, z0=1000, z1=3500,
#                 txtlist=[(9380, 1940, 'D1'), (10118, 2000, 'D2'),
#                          (10315, 1155, 'C3'), (11065, 1970, 'C4'),
#                          (10559, 2970, 'E1')])

# window_marmousi(datadir, fname, figname='MamousiVp_CW',
#                 x0=9000, x1=9800, z0=1750, z1=2250)

# get_dips(datadir, fname, x0=9100, x1=9600, z0=1600, z1=2600)

blur_image(datadir, fname, 20, 5, x0=9100, x1=9600, z0=1600, z1=2600)
