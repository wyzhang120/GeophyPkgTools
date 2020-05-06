import os
import numpy as np
from scipy.special import hankel1
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import h5py


class PltToy2dac:
    def __init__(self, datadir, freqlist, nfast, nslow, dx,
                 npml, zsrc, zrec, xsrc, xrec, z0=0, x0=0, fastz=True):
        """
        Read and plot toy2dac forward modeling seismic data, spectrum, and wavefield
        :param datadir: string,
        :param freqlist: array-like,
        :param nfast: int, n in fast dimension of model
        :param nslow: int, n in slow dimension of model
        :param dx: float, spacing of model
        :param npml: int
        :param zsrc: array-like,
        :param zrec: array-like,
        :param xsrc: float,
        :param xrec: float,
        :param z0: float, z coord at uppler left corner of model
        :param x0: float, x coord at uppler left corner of model
        :param fastz: bool,
        """
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
            self.width = (nfast - 1) * self.dx
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
        """
        Plot modeled freq domain impulse response and theoretic green's function
        :param fname:
        :param zsrc_plot:
        :param zrec_plot:
        :param vp:
        :param qp:
        :return:
        """
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
        """
        Read freq domain impluse response binary file;
        output hdf5 of freq and time domain data
        :param fname: string, freq domain impluse response binary file
        :param savename: string, hdf5 filename without extension
        :param fc: float, central freq
        :param delay_n_period: float,
        :return: hdf5 file with keys
            seismo: time domain data, source wavelet = ricker(fc, delay_n_period)
            seismo_spec: freq domain data, source wavelet = ricker(fc, delay_n_period)
            spectrum: freq domain impluse response
            delay:
            fc:
            df:
            dt:
            freqlist:
        """
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
        datafft[ind2:N-ind1 + 1, :, :] = np.conj(dataf[::-1, :, :])
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
            f.create_dataset('seismo_spec', data=datafft[ind1:ind2, :, :])
            f.create_dataset('freqlist', data=self.freqlist)
            f.create_dataset('df', data=dfreq)
            f.create_dataset('dt', data=1./dfreq/(N-1))

    def plot_seismo(self, fname, fh5, zsrc_plot, zrec_plot, fc=100, delay_n_period=10):
        """
        Plot one trace of time domain data
        :param fname:
        :param fh5:
        :param zsrc_plot: float, src depth
        :param zrec_plot: float, rec depth
        :param fc: float, central freq
        :param delay_n_period: float, num of delayed periods
        :return:
        """
        if not os.path.exists(os.path.join(self.datadir, fh5 + '.h5')):
            self.freq2time(fname, fh5, fc, delay_n_period)
        with h5py.File(os.path.join(self.datadir, fh5 + '.h5'), 'r') as f:
            seis = f['seismo'][()]
            dt = f['dt'][()]
            delay = f['delay'][()]
            fc = f['fc'][()]
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

    def plot_gather(self, fname, fh5, zsrc_plot, zrec_plot=(0, 106), t_plot=(0, 1000),
                    fc=100, delay_n_period=10, aspect=1.0, figsize=(6, 6), clip=1., interp_scalar=1):
        """
        Plot a shot gather a gray scale image
        :param fname:
        :param fh5:
        :param zsrc_plot:
        :param zrec_plot:
        :param t_plot:
        :param fc:
        :param delay_n_period:
        :param aspect:
        :param figsize:
        :param clip:
        :param interp_scalar:
        :return:
        """
        # if not os.path.exists(os.path.join(self.datadir, fh5 + '.h5')):
        self.freq2time(fname, fh5, fc, delay_n_period)
        with h5py.File(os.path.join(self.datadir, fh5 + '.h5'), 'r') as f:
            seis = f['seismo'][()]
            dt = f['dt'][()]
            delay = f['delay'][()]
        nt = seis.shape[0]
        t = (dt * np.arange(nt) - delay) * 1000
        idSrc = np.argmin(np.abs(zsrc_plot - self.zsrc))
        idRec0 = np.argmin(np.abs(zrec_plot[0] - self.zrec))
        idRec1 = np.argmin(np.abs(zrec_plot[1] - self.zrec))
        idt0 = np.argmin(np.abs(t_plot[0] - t))
        idt1 = np.argmin(np.abs(t_plot[1] - t))
        data = seis[idt0:idt1 + 1, idSrc, idRec0:idRec1 + 1]
        if interp_scalar > 1:
            data = self.interp_seis(t[idt0:idt1+1], data, interp_scalar)
        amp_max = np.max(np.abs(data))
        vmin = -clip * amp_max
        vmax = clip * amp_max
        data_plot = np.clip(data, vmin, vmax)
        ext = [self.zrec[idRec0], self.zrec[idRec1], t[idt1], t[idt0]]
        fig, ax = plt.subplots(figsize=figsize)
        img = ax.imshow(data, extent=ext, cmap='Greys', aspect=aspect, vmin=vmin, vmax=vmax)
        # polarity: black is positive
        # ax.imshow(data, cmap='Greys')
        ax.set_ylabel('t (ms)')
        ax.set_xlabel('zrec (m)')
        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes('right', size='5%', pad=0.05)
        # cbar = fig.colorbar(img, cax=cax, spacing='uniform')
        plt.tight_layout()
        plt.show()
        return fig, ax

    def interp_seis(self, t, data, scalar=5.):
        """
        Interpolate data for display
        :param t:
        :param data:
        :param scalar:
        :return:
        """
        dt0 = t[1] - t[0]
        dt = dt0 / scalar
        tplot = np.arange(t[0], t[-1] + 0.5 * dt, dt)
        ntrace = data.shape[1]
        data_out = np.zeros([len(tplot), ntrace])
        for i in range(ntrace):
            data_out[:, i] = np.interp(tplot, t, data[:, i])
        return data_out

    def plot_wiggle(self, fname, fh5, zsrc_plot, zrec_plot=(0, 106), t_plot=(0, 1000),
                    fc=100, delay_n_period=10, figsize=(6, 6), clip=0.9):
        """
        Plot seismogram as wiggles
        :param fname: string, binary file of freq domain data
        :param fh5: string, hdf5 file of time domain data
        :param zsrc_plot:
        :param zrec_plot:
        :param t_plot:
        :param fc:
        :param delay_n_period:
        :param figsize:
        :param clip:
        :return:
        """
        if not os.path.exists(os.path.join(self.datadir, fh5 + '.h5')):
            self.freq2time(fname, fh5, fc, delay_n_period)
        with h5py.File(os.path.join(self.datadir, fh5 + '.h5'), 'r') as f:
            seis = f['seismo'][()]
            dt = f['dt'][()]
            delay = f['delay'][()]
        nt = seis.shape[0]
        t = (dt * np.arange(nt) - delay) * 1000
        idSrc = np.argmin(np.abs(zsrc_plot - self.zsrc))
        idRec0 = np.argmin(np.abs(zrec_plot[0] - self.zrec))
        idRec1 = np.argmin(np.abs(zrec_plot[1] - self.zrec))
        idt0 = np.argmin(np.abs(t_plot[0] - t))
        idt1 = np.argmin(np.abs(t_plot[1] - t))
        data = seis[idt0:idt1 + 1, idSrc, idRec0:idRec1 + 1]
        fig, ax = plt.subplots(figsize=figsize)
        ntrace = idRec1 - idRec0 + 1
        offsets = np.arange(1, 1 + ntrace, 1)
        taxis = t[idt0:idt1+1]
        amp_max = np.max(data)
        scalar = 1 / amp_max
        data_plot = data * scalar
        for i in range(ntrace):
            offset = offsets[i]
            x = offset + data_plot[:, i]
            ax.plot(x, taxis, 'k-')
            ax.fill_betweenx(taxis, offset, x, where=(x > offset), color='k')
        ax.set_ylabel('t [ms]')
        ax.set_xlabel('trace NO.')
        ax.set_ylim(ax.get_ylim()[::-1])
        plt.show()
        return fig, ax

    def get_id(self, x, xarr):
        idx = np.argmin(np.abs(x - xarr))
        return idx

    def get_spec(self, fh5, ricker=True):
        key = 'seismo_spec' if ricker else 'spectrum'
        with h5py.File(os.path.join(self.datadir, fh5 + '.h5'), 'r') as f:
            trace = f[key][()]
        amp = np.absolute(trace)
        phase = np.unwrap(np.angle(trace))
        return amp, phase

    def get_spec_trace(self, fh5, zsrc_plot, zrec_plot, ricker=True):
        amp, phase = self.get_spec(fh5, ricker)
        idSrc = self.get_id(zsrc_plot, self.zsrc)
        idRec = self.get_id(zrec_plot, self.zrec)
        data_amp = amp[:, idSrc, idRec]
        data_phase = phase[:, idSrc, idRec]
        return data_amp, data_phase


def greenfunc2d(xzsrc, xzrec, freqlist, vp, qp=1000):
    xzsrc = np.array(xzsrc)
    xzrec = np.array(xzrec)
    freq = np.array(freqlist)
    r = np.sqrt(np.sum((xzsrc - xzrec)**2))
    # vp = vp * (1  + 1j/(2 * qp))
    k = 2 * np.pi * freq / vp
    g = hankel1(0, k * r) * 1j/4
    return g


class InvGroupLoader:
    def __init__(self, datadir, basename, freqlist,  nx=301, nz=106, fastz=True):
        self.datadir = datadir
        self.freqlist = freqlist
        self.basename = basename
        self.nx = nx
        self.nz = nz
        self.ngroup = len(freqlist)
        self.fastz = fastz

    def read_inv_group(self, imod):
        if self.fastz:
            n0, n1 = [self.nx, self.nz]
        else:
            n0, n1 = [self.nz, self.nx]
        inv_mod = np.zeros([self.ngroup, n0, n1], dtype=np.float32)
        for i, ifreq in enumerate(self.freqlist):
            fname = self.basename.format(imod, ifreq[0], ifreq[1], ifreq[2])
            invname = os.path.join(self.datadir, fname)
            inv_mod[i] = np.fromfile(invname, dtype=np.float32).reshape((n0, n1))
        return inv_mod

    def show_fig(self, imod, vmin=None, vmax=None, trans=True):
        imgs = self.read_inv_group(imod)
        vmin = np.min(imgs) if vmin is None else vmin
        vmax = np.max(imgs) if vmax is None else vmax
        nfreq = imgs.shape[0]
        fig, ax = plt.subplots(1, nfreq)
        for i, iax in enumerate(ax):
            img = imgs[i].T if trans else imgs[i]
            iax.imshow(img, cmap='jet_r', vmin=vmin, vmax=vmax)
        plt.show()


class CompFigs:
    def __init__(self, grp1, grp2):
        """

        :param grp1: ndarry, 2d or 3d, ref fig
        :param grp2: ndarry, 3d
        """
        self.grp1 = grp1
        self.grp2 = grp2

    def get_percent_diff(self):
        nfreq = self.grp2.shape[0]
        if len(self.grp1.shape) == 2:
            n0, n1 = self.grp1.shape
            grp1 = self.grp1.reshape(1, n0, n1)
            grp1 = np.tile(grp1, (nfreq, 1, 1))
        else:
            grp1 = self.grp1
        diff = (self.grp2 - grp1) / grp1 * 100
        return diff

    def show_diff(self, figsize, trans=True, clip=1., sym='True'):
        diff = self.get_percent_diff()
        fig, ax = plt.subplots(2, diff.shape[0],figsize=figsize)
        labels = ['m/s', '%']
        for i, img in enumerate(diff):
            img = img.T if trans else img
            img0 = self.grp2[i].T if trans else self.grp2[i]
            im0 = ax[0, i].imshow(img0, cmap='jet_r')
            if sym:
                vmax = clip * np.max(np.abs(img))
                im = ax[1, i].imshow(img, cmap='seismic', vmin=-vmax, vmax=vmax)
            else:
                im = ax[1, i].imshow(img, cmap='seismic')
            for irow, imx in enumerate([im0, im]):
                divider = make_axes_locatable(ax[irow, i])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                cbar = fig.colorbar(imx, cax=cax, spacing='uniform')
                cbar.set_label(labels[irow])
        fig.tight_layout()
        plt.show()


class FreqData:
    '''
    plot amplitude and phase of output data from TOY2DAC
    python modules needed:
      import numpy as np
      import matplotlib.pyplot as plt
    rec = [nrec, rec0, drec] drec in meters
    src = [nsrc, src0, dsrc]
    freq= [nfreq, freq0, dfreq]
    shot_p in depth
    rec_p = [starting index, max rec index, stride]
    orec_p in depth
    '''

    def __init__(self, datadir, fname, rec=(210, 1, 1), src=(105, 0, 1), freq=(299, 2, 1)):
        self.datadir = datadir
        self.fname = fname
        self.rec = rec
        self.src = src
        self.freq = freq
        fin = os.path.join(datadir, fname)
        self.data_in = np.fromfile(fin, dtype=np.complex64).reshape((freq[0], src[0], rec[0]))
        self.amp = np.absolute(self.data_in)
        self.ph = np.angle(self.data_in, deg=False)

    def add_noise(self, nsr=0.05, seed=42):
        """
        Add noise to data
        :param nsr: float, amp noise / signal
        :return:
        """
        noise_amp = nsr * np.mean(self.amp)
        data_shape = np.array(self.ph.shape)
        ndata = int(np.prod(data_shape))
        np.random.seed(seed)
        noise_ph = np.reshape(2* np.pi * np.random.rand(ndata), data_shape)
        noise_real = noise_amp * np.cos(noise_ph)
        noise_img = noise_amp * np.sin(noise_ph)
        noise = noise_real + 1j * noise_img
        return noise + self.data_in

    def show(self, shot_p, rec_p=(0, 105, 5), orec_p=1., f1_p=2, f2_p=300,
             figsize=(12, 6), grey=False, amp=None, ph=None):
        # unwrap input params
        """
        Plot selected shot
        :param shot_p: float, shot depth to be plotted
        :param rec_p: int, tuple, (min, max, step) of rec index
        :param orec_p: float, origin of rec depth
        :param f1_p: float, min freq
        :param f2_p: float, max freq
        :param figsize: tuple
        :param grey: bool, grayscale to color rec depth
        :param amp: 3d float array, amplitude
        :param ph: 3d float array, phase
        :return:
        """
        f0 = self.freq[1]
        df = self.freq[2]
        dr = self.rec[2]
        s0 = self.src[1]
        ds = self.src[2]
        pshot = int((shot_p - s0) / ds)
        prec0 = rec_p[0]
        prec1 = rec_p[1]
        stride = rec_p[2]
        r_idx = np.arange(0, prec1 - prec0, stride)
        pnrec = len(r_idx)
        freq_p = np.arange(f1_p, f2_p + df, df)
        f_idx = ((freq_p - f0) / df).astype(int)
        # color gradient
        blue = np.linspace(0.1, 0.9, stride * pnrec)
        if amp is None:
            amp = self.amp
        if ph is None:
            ph = self.ph
        ph = np.degrees(ph)
        fig, axes = plt.subplots(1, 2)
        fig.set_size_inches(figsize[0], figsize[1])
        for i in r_idx:
            if grey == False:
                axes[0].plot(freq_p, 20 * np.log10(amp[f_idx, pshot, i + prec0]), color=(blue[-i - 1], 0, blue[i]))
                axes[1].plot(freq_p, ph[f_idx, pshot, i + prec0], color=(blue[-i - 1], 0, blue[i]))

            if grey == True:
                axes[0].plot(freq_p, 20 * np.log10(amp[f_idx, pshot, i + prec0]), color=str(blue[i]))
                axes[1].plot(freq_p, ph[f_idx, pshot, i + prec0], color=str(blue[i]))

            fig.legend(np.arange(orec_p, orec_p + pnrec * dr * stride, stride * dr),
                       loc='right', title='zsrc [m]')
            axes[0].set_xlabel(r'$f (Hz)$')
            axes[1].set_xlabel(r'$f (Hz)$')
            axes[0].set_ylabel(r'$amp (dB)$')
            axes[1].set_ylabel(r'$phase(^\o)$')
            axes[1].yaxis.set_label_coords(-0.1, 0.5)
        plt.show()

    def show_noisy_data(self, shot_p, nsr=0.05):
        data = self.add_noise(nsr)
        amp = np.absolute(data)
        ph = np.angle(data, deg=False)
        self.show(shot_p, amp=amp, ph=ph)

    def save_noise_data(self, savedir, nsr=0.05, seed=42):
        if not os.path.isdir(savedir):
            os.mkdir(savedir)
        data = np.complex64(self.add_noise(nsr, seed))
        fname = '{:s}_nsr{:.2f}'.format(self.fname, nsr)
        with open(os.path.join(savedir, fname), 'wb') as f:
            data.tofile(f)