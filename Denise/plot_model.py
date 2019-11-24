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

datadir = '/project/stewart/wzhang/src/DENISE-Black-Edition/par/model'
basename = 'modelTest_{}_stage_4.bin'
nx = 500
nz = 174
dx = 20.
mod = ModLoader(datadir, basename, nx, nz, dx)
cmap = mod.read_cmap('/project/stewart/wzhang/src/DENISE-Black-Edition/par/visu', 'cmap_cm.pkl')
PltImg = PltModel((mod.vp/1000, mod.vs/1000, mod.rho/1000),
                  ('vp', 'vs', 'rho'), ('km/s', 'km/s', 'g/cc'), 0,
                  mod.height, mod.width, ytitle='Depth')
fig = PltImg.viewMulti([0, 1, 2], (3, 1), (6, 3), cmap=cmap)
plt.show()