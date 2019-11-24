import os
import numpy as np
from scipy.special import hankel1
import matplotlib.pyplot as plt
import h5py


class PltToy2dac:
    def __init__(self, datadir, freqlist, nfast, nslow, dx,
                 npml, zsrc, zrec, xsrc, xrec, z0=0, x0=0, fastz=True):
        self.datadir = datadir
        self.freqlist = np.array(freqlist)
        self.npml = npml
        self.dx = dx
        self.n0 = nslow
        self.n1 = nfast
        self.z0 = z0
        self.x0 = x0
        self.zsrc = np.array(zsrc)
        self.zrec = np.array(zrec)
        self.xsrc = xsrc
        self.xrec = xrec
        self.fastz = fastz
        if fastz:
            self.height = (nfast - 1) * self.dx
            self.width = (nslow - 1) * self.dx
        else:
            self.width= (nfast - 1) * self.dx
            self.height = (nslow - 1) * self.dx

    def read_wavefield(self, fname='wavefield'):
        n0 = self.n0 + 2 * self.npml
        n1 = self.n1 + 2 * self.npml
        nfreq = len(self.freqlist)
        with open(os.path.join(self.datadir, fname), 'rb') as f:
            wf = np.fromfile(f, dtype=np.float32).reshape([nfreq, n0, n1])
        return wf

    def plot_wavefield(self, ifreq, fname='wavefield', perc=95):
        idx = np.where(ifreq == self.freqlist)[0]
        wf = self.read_wavefield(fname)
        img = wf[idx].T if self.fastz else wf[idx]
        img = np.squeeze(img)
        img = img[self.npml:-self.npml, self.npml:-self.npml]
        vmax = np.percentile(np.abs(img), perc)
        ext = (self.x0, self.x0 + self.width,
               self.z0 + self.height, self.z0)
        fig, ax = plt.subplots()
        ax.imshow(img, extent=ext, cmap='gray', vmin=-vmax, vmax=vmax)
        plt.show()

    def read_seis(self, fname):
        seis = np.fromfile(os.path.join(self.datadir, fname), dtype=np.complex64)
        nfreq = len(self.freqlist)
        nsrc = len(self.zsrc)
        nrec = len(self.zrec)
        seis = seis.reshape([nfreq, nsrc, nrec])
        return seis

    def plot_spec(self, fname, zsrc_plot, zrec_plot, vp=3000, qp=1000):
        seis = self.read_seis(fname)
        idSrc = np.argmin(np.abs(zsrc_plot - self.zsrc))
        idRec = np.argmin(np.abs(zrec_plot - self.zrec))
        trace = seis[:, idSrc, idRec]
        amp = np.absolute(trace)
        phase = np.unwrap(np.angle(trace))
        greenfunc = greenfunc2d((self.xsrc, self.zsrc[idSrc]),
                                (self.xrec, self.zrec[idRec]),
                                self.freqlist, vp, qp)
        amp_green = np.absolute(greenfunc)
        phase_green = np.unwrap(np.angle(greenfunc))
        fig, ax = plt.subplots(1, 2)
        ax[0].plot(self.freqlist, amp, 'b', label='Modeled')
        ax[0].plot(self.freqlist, amp_green, 'r', label='Theoretic')
        ax[0].set_xlabel('Freq [Hz]')
        ax[0].set_ylabel('Amp')
        ax[0].legend()
        ax[1].plot(self.freqlist, phase, 'b')
        ax[1].plot(self.freqlist, phase_green, 'r')
        ax[1].set_xlabel('Freq [Hz]')
        ax[1].set_ylabel('Phase [rad]')
        plt.show()

    def freq2time(self, fname, savename, fc, delay_n_period=10):
        dataf = self.read_seis(fname)
        _, nsrc, nrec = dataf.shape
        fmax = self.freqlist.max()
        dfreq = self.freqlist[1] - self.freqlist[0]
        Nhalf = int(fmax / dfreq)
        N = 1 + 2 * Nhalf
        f = dfreq * np.roll(np.arange(-Nhalf, Nhalf + 1), -Nhalf)
        ind1 = int(self.freqlist[0] / dfreq)
        ind2 = Nhalf + 1
        datafft = np.zeros((N, nsrc, nrec), dtype=np.complex64)
        datafft[ind1:ind2, :, :] = dataf
        datafft[ind2:-ind1 + 1, :, :] = np.conj(dataf[::-1, :, :])
        rconst = np.sqrt(np.pi)
        delay = delay_n_period / fc
        ricker_delay = 2. / rconst * f ** 2 / fc ** 3 * np.exp(
            -(f / fc) ** 2 + 1j * 2 * np.pi * f * delay)
        for i in range(nsrc):
            for j in range(nrec):
                datafft[:, i, j] = ricker_delay * datafft[:, i, j]
        datat = -np.real(1./N * dfreq * np.fft.fftn(datafft, s=[N], axes=[0]))
        with h5py.File(os.path.join(self.datadir, savename + '.h5'), 'w') as f:
            f.create_dataset('seismo', data=datat)
            f.create_dataset('fc', data=fc)
            f.create_dataset('delay', data=delay)
            f.create_dataset('spectrum', data=dataf)
            f.create_dataset('freqlist', data=self.freqlist)
            f.create_dataset('df', data=dfreq)
            f.create_dataset('dt', data=1./dfreq/(N-1))

    def plot_seismo(self, fname, fh5, zsrc_plot, zrec_plot, fc=100, delay_n_period=10):
        if not os.path.exists(os.path.join(self.datadir, fh5 + '.h5')):
            self.freq2time(fname, fh5, fc, delay_n_period)
        with h5py.File(os.path.join(self.datadir, fh5 + '.h5'), 'r') as f:
            seis = f['seismo'].value
            dt = f['dt'].value
            delay = f['delay'].value
            fc = f['fc'].value
        nt = seis.shape[0]
        ntPlot = nt
        t = dt * np.arange(ntPlot) * 1000
        idSrc = np.argmin(np.abs(zsrc_plot - self.zsrc))
        idRec = np.argmin(np.abs(zrec_plot - self.zrec))
        trace = seis[:ntPlot, idSrc, idRec]
        fig, ax = plt.subplots()
        ax.plot(t, trace)
        ax.set_xlabel('t [ms]')
        ax.set_title('zsrc = {:g} [m]\n'
                     ' delay = {:g} [ms], fc = {:g} Hz'.format(
            self.zsrc[idSrc], delay * 1000, fc))
        print('zsrc = {:g} [m], xsrc = {:g} [m] \n'
              'zrec = {:g} [m], xrec = {:g} [m]'.format(
               self.zsrc[idSrc], self.xsrc, self.zrec[idRec], self.xrec))
        plt.show()



def greenfunc2d(xzsrc, xzrec, freqlist, vp, qp=1000):
    xzsrc = np.array(xzsrc)
    xzrec = np.array(xzrec)
    freq = np.array(freqlist)
    r = np.sqrt(np.sum((xzsrc - xzrec)**2))
    vp = vp * (1 - 1j/(2 * qp))
    k = 2 * np.pi * freq / vp
    g = hankel1(0, k * r) * 1j/4
    return g