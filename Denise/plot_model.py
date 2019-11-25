import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PkgTools.Denise.utils_denise import ModLoader, PltModel

# datadir = '/project/stewart/wzhang/src/DENISE-Black-Edition/par_fdtest/model'
# fname = 'CW_fdtest.vp'
# nz = 840
# nx = 440
# dx = 1.25
# ext = (0, (nx-1) * dx, (nz-1)*dx, 0)
# with open(os.path.join(datadir, fname), 'rb') as f:
#     data = np.fromfile(f, dtype=np.float32).reshape([nx, nz])
# fig, ax = plt.subplots()
# img = ax.imshow(data.T, extent=ext, cmap='jet')
# divider = make_axes_locatable(ax)
# cax = divider.append_axes('right', size='5%', pad=0.05)
# cbar = fig.colorbar(img, cax=cax, spacing='uniform')
# plt.show()

# datadir = '/project/stewart/wzhang/src/DENISE-Black-Edition/par/model'
datadir = 'D:\Geophysics\Project\Marmousi\FWI_Denise'
basename_inv = 'modelTest_{}_stage_4.bin'
basename_true = 'marmousi_II_marine.{}'
basename_init = 'marmousi_II_start_1D.{}'
nx = 500
nz = 174
dx = 20.
mod_inv = ModLoader(datadir, basename_inv, nx, nz, dx)
mod_true = ModLoader(datadir, basename_true, nx, nz, dx)
mod_init = ModLoader(datadir, basename_init, nx, nz, dx)
# cmap = mod_inv.read_cmap('/project/stewart/wzhang/src/DENISE-Black-Edition/par/visu', 'cmap_cm.pkl')
cmap = 'jet'
titles = [j + i for i in ('_inv', '_true', '_init') for j in ('vp', 'vs', 'rho')]
PltImg = PltModel((mod_inv.vp/1000, mod_inv.vs/1000, mod_inv.rho/1000,
                   mod_true.vp/1000, mod_true.vs/1000, mod_true.rho/1000,
                   mod_init.vp/1000, mod_init.vs/1000, mod_init.rho/1000,),
                  titles, ['km/s', 'km/s', 'g/cc'] * 3, 0,
                  mod_inv.height, mod_inv.width, padDist=(0, 0, -3500, 0), ytitle='Depth')
fig = PltImg.viewMulti(np.arange(3, dtype=np.int32), (3, 1), (6, 1.5), cmap=cmap)
fig.savefig(os.path.join(datadir, 'Marm_Denise_FWI.pdf'))
fig = PltImg.viewMulti(np.arange(3, 6, dtype=np.int32), (3, 1), (6, 1.5), cmap=cmap)
fig.savefig(os.path.join(datadir, 'Marm_Denise_True.pdf'))
fig = PltImg.viewMulti(np.arange(6, 9, dtype=np.int32), (3, 1), (6, 1.5), cmap=cmap)
fig.savefig(os.path.join(datadir, 'Marm_Denise_Init.pdf'))
fig = PltImg.viewMulti((3, 6, 0), (3, 1), (6, 1.5), cmap=cmap, clim=(1, 5.5))
fig.savefig(os.path.join(datadir, 'Marm_Denise_Vp_Comp.pdf'))
