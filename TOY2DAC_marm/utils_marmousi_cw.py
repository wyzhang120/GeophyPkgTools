import os
import numpy as np
import segyio


def crop_mamousi(datadir, fname, x0=9100, x1=9600, z0=1500, z1=2500, dx=1.25):
    """
    Crop a subset from the segy file of Mamousi model
    :param datadir: string
    :param fname: string
    :param x0: float, xmin
    :param x1: float, xmax
    :param z0: float, zmin
    :param z1: float, zmax
    :param dx: float, grid spacing
    :return:
        mod: 2d array, float, shape = (nx, nz)
    """
    with segyio.open(os.path.join(datadir, fname), mode='r') as f:
        vp = f.xline
        idx0 = int(np.round(x0 / dx))
        nx = int(np.round((x1 - x0) / dx) + 1)
        idz0 = int(np.round(z0 / dx))
        nz = int(np.round(z1 - z0) / dx + 1)
        mod = vp[0][idx0:idx0 + nx, idz0:idz0 + nz]
    return mod


class AcqCw:
    def __init__(self, srcpar, recpar):
        self.src = srcpar
        self.rec = recpar

    def getAcqDict(self):
        srcpar = self.src
        recpar = self.rec
        zsrc = np.arange(srcpar['zmin'], srcpar['zmax'], srcpar['dz'])
        zrec = np.arange(recpar['zmin'], recpar['zmax'], recpar['dz'])
        xsrc = srcpar['x0']
        xrec = recpar['x0']
        outDict = {'xsrc': xsrc, 'zsrc': zsrc, 'xrec': xrec, 'zrec': zrec}
        return outDict
