import os
import cv2
import numpy as np
from sympy.ntheory import factorint
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle
import pandas as pd
from skimage.transform import resize


class AcqCw2:
    def __init__(self, srcpar, recpar):
        """
        Prepare source and receiver coord inputs for Denise
        :param srcpar: dict, keys =
                       zmin, zmax, dz, x0: float
                       fname : string, filename
                       srctype: int, 1=explosive, 2=point force in x, 3=point force in z, 4=custom directive force
                       fc: float, central freq
                       amp: float, amplitude
        :param recpar:
        """
        self.src = srcpar
        self.rec = recpar
        zsrc = np.arange(srcpar['zmin'], srcpar['zmax'], srcpar['dz'])
        zrec = np.arange(recpar['zmin'], recpar['zmax'], recpar['dz'])
        xsrc = srcpar['x0']
        xrec = recpar['x0']
        outDict = {'xsrc': xsrc, 'zsrc': zsrc, 'xrec': xrec, 'zrec': zrec}
        self.acqdict = outDict

    def write_acq(self, basedir):
        self.write_src(basedir)
        self.write_rec(basedir)

    def write_src(self, basedir):
        nsrc = len(self.acqdict['zsrc'])
        xsrc = self.acqdict['xsrc']
        zsrc = self.acqdict['zsrc']
        with open(os.path.join(basedir, self.src['fname']), 'w') as f:
            f.write('{:d}\n'.format(nsrc))
            # XSRC ZSRC YSRC TD FC AMP ANGLE QUELLTYP (NSRC lines)
            for i in range(nsrc):
                f.write('{:<16g} {:<8g} {:<16g} {:<8g} {:<8g} {:<8g} {:<8g} {:<8d}\n'.format(
                    xsrc, 0, zsrc[i], 0, self.src['fc'], self.src['amp'], 0., self.src['srctype']
                ))

    def write_rec(self, basedir):
        nrec = len(self.acqdict['zrec'])
        xrec = self.acqdict['xrec']
        zrec = self.acqdict['zrec']
        with open(os.path.join(basedir, self.rec['fname'] + '.dat'), 'w') as f:
            for i in range(nrec):
                f.write('{:<16g} {:<16g}\n'.format(xrec, zrec[i]))


def print_factors(nx, nz):
    """
    factorize nx and nx to determine decomposition for DENSIE input file
    :param nx:
    :param nz:
    :return:
    """
    factornx = factorint(nx)
    factornxStr = ', '.join(['{:d}'.format(i) for i in factornx.keys()])
    powernxStr = ', '.join(['{:d}'.format(val) for _, val in factornx.items()])
    factornz = factorint(nz)
    factornzStr = ', '.join(['{:d}'.format(i) for i in factornz.keys()])
    powernzStr = ', '.join(['{:d}'.format(val) for _, val in factornz.items()])
    print('nx = {:d}, factors = ({:s}), power = ({:s}), \n'
          'nz = {:d}, factors = ({:s}), power = ({:s})'.format(
        nx, factornxStr, powernxStr, nz, factornzStr, powernzStr, ))


def write_mfile(mfile, dict_mfile, basedir):
    """
    Write model file : from dict to binary files
    :param mfile: string, file name
    :param dict_mfile: dict, keys = [vp, vs, rho], each is a ndarray
    :param basedir: string
    :return:
    """
    for key, data in dict_mfile.items():
        with open(os.path.join(basedir, '{:s}.{:s}'.format(mfile, key)), 'wb') as f:
            data.tofile(f)


class ModLoader:
    def __init__(self, datadir, basename, nx, nz, dx,
                 keys=('vp', 'vs', 'rho'), fastz=True, trans=True):
        self.datadir = datadir
        self.basename = basename
        self.nx = nx
        self.nz = nz
        self.dx = dx
        self.height = (nz - 1) * dx
        self.width = (nx - 1) * dx
        self.fastz = fastz
        attrs = ('vp', 'vs', 'rho')
        for ikey, iattr in zip(keys, attrs):
            if ikey is not None:
                self.__setattr__(iattr, self.readmod(ikey, trans=trans))
        # self.vp = self.readmod(keys[0])
        # self.vs = self.readmod('vs')
        # self.rho = self.readmod('rho')

    def readmod(self, key, byteorder='<', trans=True):
        data_type = np.dtype('float32').newbyteorder(byteorder)
        shape = (self.nx, self.nz) if self.fastz else (self.nz, self.nx)
        with open(os.path.join(self.datadir, self.basename.format(key)), 'rb') as f:
            mod = np.fromfile(f, dtype=data_type).reshape(shape)
        mod = mod.T if trans else mod
        return mod

    def read_cmap(self, datadir, fname):
        with open(os.path.join(datadir, fname), 'rb') as f:
            cmap = pickle.load(f)
        return cmap

    def resize(self, dx, attr):
        s = self.dx / dx
        arr = self.__getattribute__(attr)
        size = np.int32((np.array(arr.shape) - 1) * s + 1)
        arr_out = resize(arr, size, preserve_range=True, mode='constant')
        return arr_out


def resize_mod(modx, dxOut, dx0):
    s = dx0 / dxOut
    size = np.int32((np.array(modx.shape) - 1) * s + 1)
    arr_out = resize(modx, size, preserve_range=True, mode='constant')
    return arr_out

def workflow_parser(indir, in_name, outdir, out_name):
    df = pd.read_csv(os.path.join(indir, in_name), sep='\s+', header=0)
    df.to_csv(os.path.join(outdir, out_name))


class PltModel:
    def __init__(self, img, tt, tbar, z0, height, width,
                 padDist=(0, 0, 0, 0), unit='m',
                 ucvtOverwrite=False, ucvt=1., xtitle='x', ytitle='TVDss'):
        """
        Plot a 2d array
        :param img: list of 2d array
        :param tt: list of string, title
        :param tbar: list of string, title of colorbar
        :param npad:  int or list [top, bottom, left, right], padding of 2D model params
        """
        nimg = len(img)
        ntt = len(tt)
        ntbar = len(tbar)
        if not (nimg == ntt and ntt == ntbar):
            raise ValueError("The number of image arrays, titles and color bar titles must be the same.")
        self.padDist = padDist
        self.img = img
        self.tt = tt
        self.tbar = tbar
        self.z0 = z0
        self.height = height
        self.width = width
        self.unit = unit
        self.ft2m = 0.3048
        self.xtt = xtitle
        self.ytt = ytitle
        if unit == 'm':
            self.ucvt = 1.
        elif unit == 'ft':
            self.ucvt = 3.28084
        else:
            raise ValueError('Length unit not recognized')
        if ucvtOverwrite:
            self.ucvt = ucvt


    def __draw__(self, idx, fig, ax, cmap,
                 pltAcq, zsrc, zrec, srcSym, recSym, padCbar, hpad,
                 clim=None, hWells=None, ctitle='k', mksizeRec=10):
        """
        Plot a 2D image for given figure and ax
        :param fig: figure object
        :param ax: ax object
        :param cmap: string, color map
        :param ctitle: string, color of the ax title
        :return: fig, ax
        """
        padDist = np.array(self.padDist) * self.ucvt
        zmin = self.z0 * self.ucvt
        zmax = (self.z0 + self.height) * self.ucvt
        xmin = -1 *self.padDist[2] * self.ucvt
        xmax = self.width * self.ucvt + xmin
        ext = (xmin, xmax, zmax, zmin)
        if clim is not None:
            im = ax.imshow(self.img[idx], cmap=cmap, extent=ext,
                            vmin=clim[0], vmax=clim[1])
        else:
            im = ax.imshow(self.img[idx], cmap=cmap, extent=ext)
        if pltAcq:
            xsrc = np.zeros(len(zsrc))
            xrec = np.ones(len(zrec)) * (xmax - padDist[2])
            ax.plot(xsrc, zsrc, srcSym, zorder=10, clip_on=False)
            ax.plot(xrec, zrec, recSym, zorder=10, clip_on=False, markersize=mksizeRec)
        if hWells is not None:
            for iwell in hWells.keys():
                if self.unit == 'ft':
                    xz = hWells[iwell].xzSect
                elif self.unit == 'm':
                    xz = hWells[iwell].xzSect * self.ft2m
                else:
                    raise ValueError('Length unit not recognized')
                ax.scatter(xz[0], xz[1], color='k', s=10, alpha=0.5)
                ax.annotate(iwell, (xz[0] - 200, xz[1]), color='k',
                            fontstyle='italic', fontweight='demi')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=padCbar)
        cbar = fig.colorbar(im, cax=cax)
        if self.tbar[idx] is not None:
            cbar.set_label(self.tbar[idx])
        ax.set_xlabel('{:s} ({:s})'.format(self.xtt, self.unit))
        ax.set_ylabel('{:s} ({:s})'.format(self.ytt, self.unit))

        if self.tt[idx] is not None:
            ax.set_title(self.tt[idx], color=ctitle)
        plt.tight_layout(h_pad=hpad)
        return fig, ax

    def view(self, idx=0, figsize=(6, 6), cmap='jet',
             pltAcq=False, zsrcPar=None, zrecPar=None,
             srcSym='r*', recSym='g<', padCbar=0.05, hpad=1.5,
             clim=None, returnAx=False):
        """
        Show figure
        :param figsize: list, figure size, [width, height]
        :param cmap: string, color map
        :param pltAcq: bool, whether to plot sources and receivers
        :param zsrcPar: list of floats, zmin, zmax, dz of sources, unit=ft
        :param zrecPar: list of floats, zmin, zmax, dz of receivers, unit=ft
        :param srcSym: string, symbols of sources in figure
        :param recSym: string, symbols of receivers in figure
        :return:
        """
        if pltAcq:
            if zsrcPar is None or zrecPar is None:
                raise ValueError('pltAcq is True, src and rec depth params must be provided')
            zsrc = (self.z0 + zsrcPar) * self.ucvt
            zrec = (self.z0 + zrecPar) * self.ucvt
        else:
            zsrc = None
            zrec = None

        fig, ax = plt.subplots()
        fig.set_size_inches(figsize[0], figsize[1])
        fig, ax = self.__draw__(idx, fig, ax, cmap, pltAcq, zsrc, zrec,
                                srcSym, recSym, padCbar, hpad, clim)
        plt.show()
        if returnAx:
            return fig, ax
        else:
            return fig

    def viewMulti(self, idxlist, axShape, ifigsize, cmap='jet',
                  pltAcq=False, zsrcPar=None, zrecPar=None,
                  srcSym='r*', recSym='g<', padCbar=0.05, hpad=1.5,
                  clim=None, hWells=None, ctitle=None, returnAx=False,
                  mksizeRec=10):
        if pltAcq:
            if zsrcPar is None or zrecPar is None:
                raise ValueError('pltAcq is True, src and rec depth params must be provided')
            zsrc = (self.z0 + zsrcPar) * self.ucvt
            zrec = (self.z0 + zrecPar) * self.ucvt
        else:
            zsrc = None
            zrec = None
        nr, nc = axShape
        if nr * nc != len(idxlist):
            raise ValueError(" Number of inputs images doesn't match the shape of ax.")
        fig, ax = plt.subplots(nr, nc)
        ax = ax.reshape([nr, nc])
        fig.set_size_inches(ifigsize[0] * nc, ifigsize[1] * nr + nr * hpad)
        if ctitle is None:
            ctitle = ['k'] * len(self.tbar)
        for i in range(nr):
            for j in range(nc):
                idx = idxlist[i * nc + j]
                fig, ax[i, j] = self.__draw__(idx, fig, ax[i, j], cmap, pltAcq,
                                              zsrc, zrec, srcSym, recSym, padCbar, hpad,
                                              clim, hWells, ctitle[idx], mksizeRec)
        plt.show()
        if returnAx:
            return fig, ax
        else:
            return fig

    def save(self, fig, fname, ftype='.pdf'):
        """
        Save figure
        :param fig: figure object
        :param fname: string, name of figure
        :param ftype: string, type of file
        :return:
        """
        fig.savefig(fname+ftype)
        plt.close(fig)


class LocateDeniseMod:
    def __init__(self, par0, par, segydir, segyvp):
        vp_denise = self.resizex(par0, par)
        nz, nx = vp_denise.shape if par0['trans'] else vp_denise.shape[::-1]
        width = (nx - 1) * par['dx']
        height = (nz - 1) * par['dx']
        self.resize_denise = vp_denise
        self.width = width
        self.height = height
        self.nz = nz
        self.nx = nx
        self.segydir = segydir
        self.segyvp = segyvp
        self.par = par
        self.par0 = par0

    def resizex(self, par0, par1):
        loader1 = ModLoader(par0['dir'], par0['basename'], par0['nx'], par0['nz'],
                            par0['dx'], fastz=par0['fastz'])
        vp1 = loader1.readmod('vp', trans=par0['trans'])
        scaler = par0['dx'] / par1['dx']
        size2 = np.int32((np.array(vp1.shape) - 1) * scaler + 1)
        vp2 = resize(vp1, size2, preserve_range=True, mode='constant')
        return vp2

    def locate_template(self, img, template):
        cxr = cv2.matchTemplate(np.float32(img), np.float32(template),
                                method=cv2.TM_CCORR_NORMED)
        ind = np.unravel_index(np.argmax(cxr), cxr.shape)
        return ind

    def locate_mod(self, xsearch, zsearch):
        from TOY2DAC_marm.utils_marmousi_cw import crop_mamousi
        vp_segy = crop_mamousi(self.segydir, self.segyvp, x0=xsearch[0], x1=xsearch[1],
                               z0=zsearch[0], z1=zsearch[1], dx=self.par['dx'])
        ind_align = self.locate_template(vp_segy, self.resize_denise)
        offset = np.array(ind_align) * self.par['dx'] + np.array([xsearch[0], zsearch[0]])
        return ind_align, offset


class PltSeisDenise:
    """
    Plot seismic gather in binary format output from Denise
    """
    def __init__(self, par):
        self.datadir = par['datadir']
        self.basename = par['basename']
        self.nt = par['nt']
        self.ntrace = par['zrec'].shape[0]
        self.dt = par['dt'] # ms
        self.fc = par['fc']
        self.delay_n_period = par['delay_n_period']
        self.zrec = par['zrec']
        self.zsrc = par['zsrc']

    def read_gather(self, imod, zshot):
        ishot = int((zshot - self.zsrc[0]) / (self.zsrc[1] - self.zsrc[0]))
        fname = os.path.join(self.datadir, self.basename.format(imod, ishot))
        seis = np.fromfile(fname, dtype=np.float32).reshape((self.ntrace, self.nt))
        return seis

    def plot_gather(self, imod, zshot, tlim, nsr=0,
                    basename='marm{:d}_shot{:d}m_nsr{:d}.pdf', aspect=1.0, seis=None):
        ishot = int((zshot - self.zsrc[0]) / (self.zsrc[1] - self.zsrc[0]))
        if seis is None:
            seis = self.read_gather(imod, ishot)
        tlim = np.array(tlim)
        t0 = -self.delay_n_period / self.fc
        tlim = np.clip(tlim, 0, t0 + self.nt * self.dt)
        idt = np.int32(tlim / self.dt)
        tplot = self.dt * idt
        data = seis[:, idt[0] : idt[1] + 1]
        ext = (self.zrec[0], self.zrec[-1], tplot[1], tplot[0])
        fig, ax = plt.subplots()
        im = ax.imshow(data.T, cmap='Greys', extent=ext, aspect=aspect)
        # plt.colorbar(im)
        ax.set_ylabel('t (ms)')
        ax.set_xlabel('zrec (m)')
        plt.tight_layout()
        fname = basename.format(imod, int(self.zsrc[ishot]), nsr)
        fig.savefig(os.path.join(self.datadir, fname))
        plt.show()
        plt.close(fig)

    def add_noise(self, imod, zshot, nsrlist=(0, 1., 3.), tlim=(50, 200),
                  ztrace=200, rseed=39, colors=('r', 'k', 'b'),
                  basename='marm{:d}_shot{:d}m_tr{:d}_spec.pdf'):
        seist = self.read_gather(imod, zshot)
        ishot = int((zshot - self.zsrc[0]) / (self.zsrc[1] - self.zsrc[0]))
        seisf = np.fft.fft(seist, axis=1) / self.nt
        amp = np.abs(seisf)
        dt = self.dt/1000
        df = 1./ dt / self.nt
        f = df * np.arange(0, self.nt)
        fc = self.fc
        idf1 = np.nonzero(f > 300)[0][0]
        rickerf0 = 2 * f**2 / (np.sqrt(np.pi) * fc**3) * np.exp(-(f/fc)**2)
        rickerf = np.copy(rickerf0)
        rickerf[0] = rickerf[1]
        rickerf = rickerf.reshape([1, -1])
        rickerf = np.tile(rickerf, (self.ntrace, 1))
        imp_amp = np.zeros(amp.shape)
        imp_amp[:, :idf1] = amp[:, :idf1]/rickerf[:, :idf1]
        seis_amp = [amp]
        self.plot_gather(imod, zshot, tlim, 0, aspect=1.0, seis=seist)
        np.random.seed(rseed)
        delay = self.delay_n_period / dt
        data_shape = amp.shape
        ndata = int(np.prod(data_shape))
        for nsr in nsrlist[1:]:
            noise_amp = nsr * np.mean(imp_amp[:, 1:idf1])
            noise_phs = np.reshape(2 * np.pi * np.random.rand(ndata), data_shape)
            noise_real = noise_amp * np.cos(noise_phs)
            noise_img = noise_amp * np.sin(noise_phs)
            # noise = (noise_real + 1j * noise_img)
            noise = (noise_real + 1j * noise_img) * rickerf0 * np.exp(-1j * f * delay)
            seisf_noisy = noise + seisf
            seist_noisy = np.real(np.fft.ifft(seisf_noisy, axis=1) * self.nt)
            self.plot_gather(imod, zshot, tlim, int(nsr), aspect=1.0, seis=seist_noisy)
            seisf_amp = np.abs(seisf_noisy)
            seis_amp.append(seisf_amp)
        itrace = int((ztrace - self.zrec[0]) / (self.zrec[1] - self.zrec[0]))
        ztrace = self.zrec[itrace]
        fig, ax = plt.subplots()
        for i, ampx in enumerate(seis_amp):
            ax.plot(f[1:idf1], ampx[itrace, 1:idf1], colors[i],
                    label='nsr = {:d}'.format(int(nsrlist[i])))
            ax.legend()
            ax.set_xlabel('f (Hz)')
            ax.set_ylabel('Amp')
            ax.set_xlim(0, 300)
            ax.set_xlim(0, 300)
        plt.tight_layout()
        fname = basename.format(imod, int(self.zsrc[ishot]), int(ztrace))
        fig.savefig(os.path.join(self.datadir, fname))
        plt.show()
        plt.close(fig)
        fig, ax = plt.subplots()
        for i, ampx in enumerate(seis_amp):
            ax.plot(f[1:idf1], np.mean(ampx[:, 1:idf1], axis=0), colors[i],
                    label='nsr = {:d}'.format(int(nsrlist[i])))
            ax.legend()
            ax.set_xlabel('f (Hz)')
            ax.set_ylabel('Amp')
            ax.set_xlim(0, 300)
            ax.set_xlim(0, 300)
        plt.tight_layout()
        fname = 'marm{:d}_shot{:d}m_mean_spec.pdf'.format(imod, int(self.zsrc[ishot]))
        fig.savefig(os.path.join(self.datadir, fname))
        plt.show()


