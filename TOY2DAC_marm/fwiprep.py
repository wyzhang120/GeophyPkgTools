import os
import numpy as np
from datetime import datetime
from subprocess import check_call, Popen
import time
import multiprocessing
from shutil import copyfile
from skimage.filters import gaussian
import cv2 as cv
import matplotlib.pyplot as plt


def acq(pars, parr, dzsrc=None, dzrcv=None):
    n1 = pars['nshots'] * (parr['nrV'] + 1)
    n2 = 5
    if dzsrc is None:
        dzsrc = pars['jsV'] * pars['dz']
    if dzrcv is None:
        dzrcv = parr['jrV'] * pars['dz']
    rmask = (np.arange(0, n1, 1) % (parr['nrV'] + 1)) != 0
    smask = (np.arange(0, n1, 1) % (parr['nrV'] + 1)) == 0
    acq = np.zeros((n1, n2), dtype=np.float32)
    # source x, z
    acq[smask, 1] = pars['xsrc']
    acq[smask, 0] = np.linspace(pars['osV'],
                                pars['osV'] + dzsrc * (pars['nshots'] - 1),
                                pars['nshots'])

    # receiver x,z
    acq[rmask, 1] = parr['xrec']
    acq[rmask, n2 - 1] = np.tile(np.ones(parr['nrV'], np.int8), pars['nshots'])
    zrcv = np.linspace(parr['orV'],
                       parr['orV'] + dzrcv * (parr['nrV'] - 1),
                       parr['nrV'])
    acq[rmask, 0] = np.tile(zrcv, pars['nshots'])
    facq= os.path.join(pars['inv_path'], parr['facq'])
    with open(facq, 'w') as f:
        np.savetxt(f, acq,
                   fmt=['%g', '%g', '%g', '%g', '%d'],
                   delimiter=' ')


def write_init_mod(vp, par):
    vp_init = np.copy(vp)
    for i in range(par['repeat']):
        vp_init = gaussian(vp_init, par['sigma_init_mod'])
    vp_init = np.float32(vp_init)
    finit = os.path.join(par['inv_path'], par['fvpinit0'])
    with open(finit, 'w') as f:
        vp_init.tofile(f)


def freqm(par):
    freqlist = np.arange(par['freq0'], par['freq-1'], par['freq_step'])
    nfreq = len(freqlist)
    with open(os.path.join(par['inv_path'], par['ffreq_man']), 'w') as f:
        f.write('%d\n' % nfreq)
        freqlist.tofile(f, sep=' ', format='%g')
        f.write('\n')


def write_vp_rho_qp(par, vp, rho):
    sigma = par['sigma_true_mod']
    vp = np.float32(gaussian(vp, sigma))
    rho = np.float32(gaussian(rho, sigma))
    qp = np.ones(vp.shape, dtype=np.float32)
    data = (vp, rho, qp)
    fnames = (par['fvp'], par['frho'], par['fqp'])
    for ifile, idata in zip(fnames, data):
        with open(os.path.join(par['inv_path'], ifile), 'wb') as f:
            idata.tofile(f)


def bathym(par):
    seafloor=par['seafloor']*np.ones(par['nx'], np.float32)
    with open(os.path.join(par['inv_path'], par['fbath']),'wb') as f:
         seafloor.tofile(f)


def data_weight_file(par):
    w = par['dataw']*np.ones(par['ndataw'],np.float32)
    with open(os.path.join(par['inv_path'], par['fdataw']),'w') as f:
         np.savetxt(f,w,fmt='%g',delimiter=' ')


def data_weight_voffset(par, srcpar, recpar):
    """
    Assign data offset based on vertical src-rec offset in crosswell geometry
    :param par: dict
    :par h: float, horizontal offset between src and rec
    :return:
    """

    ds = par['ddataw']
    h = abs(srcpar['x0'] - recpar['x0'])
    inv_offset = np.sqrt(h**2 + par['inv_z_offset']**2)
    max_z_offset = max((abs(srcpar['zmin'] - recpar['zmax']),
                        abs(srcpar['zmax'] - recpar['zmin'])))
    max_offset = np.sqrt(h**2 + max_z_offset**2)
    n = int(max_offset / ds + 1)
    n1 = int(inv_offset / ds)
    w = np.zeros(n, dtype=np.float32)
    w[:n1] = 1.
    with open(os.path.join(par['inv_path'], par['fdataw']),'w') as f:
         np.savetxt(f, w, fmt='%g',delimiter=' ')
    return n


def data_window(fname, freqlist, fshape, fout, freqmng, f0=2, df=1):
    data=np.fromfile(fname,dtype=np.complex64)
    data=np.reshape(data, fshape)
    fidx = ((freqlist - f0)/df).astype(np.int32)
    dataw = data[fidx, :, :]
    dataw.tofile(fout)
    nfreq = len(freqlist)
    with open(freqmng, 'w') as f:
        f.write('%d\n'%nfreq)
        freqlist.tofile(f, sep=' ', format='%g')
        f.write('\n')


def data_window1(fname, freqlist, fshape, fout, freqmng, nrec1=105, f0=2, df=1):
    data=np.fromfile(fname,dtype=np.complex64)
    data=np.reshape(data, fshape)
    fidx = ((freqlist - f0)/df).astype(np.int32)
    dataw = data[fidx, :, :nrec1]
    dataw.tofile(fout)
    nfreq = len(freqlist)
    with open(freqmng, 'w') as f:
        f.write('%d\n'%nfreq)
        freqlist.tofile(f, sep=' ', format='%g')
        f.write('\n')


def datasep(fname, freqlist, fshape, fout, freqmng, nrec1=105, f0=2, df=1):
    """

    :param fname:
    :param freqlist:
    :param fshape:
    :param fout: output file base name
    :param freqmng: freq_management file name
    :param nrec1:
    :param f0:
    :param df:
    :return:
    """
    data=np.fromfile(fname,dtype=np.complex64)
    data=np.reshape(data, fshape)
    fidx = ((freqlist - f0)/df).astype(np.int32)
    dataw1 = data[fidx, :, :nrec1]
    dataw1.tofile(fout+'r')
    dataw1 = data[fidx, :, nrec1:]
    dataw1.tofile(fout + 'l')
    nfreq = len(freqlist)
    with open(freqmng, 'w') as f:
        f.write('%d\n' %nfreq)
        freqlist.tofile(f, sep=' ', format='%g')
        f.write('\n')


def t2dac_in_inv(par, facq):
    fin=os.path.join(par['inv_path'], par['f2dac'])
    with open(fin,'w') as f:
        f.write('''%d  ! mode of the code 0-> modeling, 1-> FWI\n
        %d ! forward modeling tool 1->fdfd iso, 2-> fdfd aniso\n
        %s ! acquistion file name''' %(par['toymode'], par['iso'], facq))		


def mumps_in(par):
    with open(os.path.join(par['inv_path'], par['fmumps']),'w') as f:
         f.write('%(icntl_7)d\n%(icntl_14)d\n%(icntl_23)d\n%(keep_84)d\n' %par)


def fd_in_iso(par):
    fname = os.path.join(par['inv_path'], par['ffd'])
    with open(fname,'w') as f:
        f.write('%(nz)d %(nx)d\n'%par)
        f.write('%(dx)g\n'%par)
        if par['iso']==1:
            f.write('%(fvp)s %(fqp)s %(frho)s\n' %par)
        elif par['iso']==2:
            f.write('%(fvp)s %(fqp)s %(frho)s %(feps)s %(fdel)s %(ftheta)s\n' %par)
        f.write('%(pml_coef)g %(npml)d\n' %par)
        f.write('%(Hicks_interp)d\n' %par)
        f.write('%(free_surf)d\n' %par)
        f.write('%(srctype)d %(rcvtype)d\n' %par)
        f.write('%(slaplace)g\n' %par)


def fd_in_iso_inv(par):
    fname = os.path.join(par['inv_path'], par['ffd'])
    with open(fname,'w') as f:
        f.write('%(nz)d %(nx)d\n'%par)
        f.write('%(dx)g\n'%par)
        if par['iso']==1:
            f.write('%(fvpinit)s %(qp_init)s %(rho_init)s\n' %par)
        elif par['iso']==2:
            f.write('%(fvpinit)s %(qp_init)s %(rho_init)s %(eps_init)s %(del_init)s %(theta_init)s\n' %par)
        f.write('%(pml_coef)g %(npml)d\n' %par)
        f.write('%(Hicks_interp)d\n' %par)
        f.write('%(free_surf)d\n' %par)
        f.write('%(srctype)d %(rcvtype)d\n' %par)
        f.write('%(slaplace)g\n' %par)


def fwi_in(par, fdata):
    fname = os.path.join(par['inv_path'], par['ffwi'])
    with open(fname,'w') as f:
        f.write('%s     !name of obs data\n'%fdata)
        f.write('%(family)d       !family\n'%par)
        f.write(' '.join(str('%d')%i for i in par['invpars']))
        f.write('     !npar & inverted para\n')
        f.write('%(src_est)d        ! src estimation\n'%par)
        f.write('%(fbath)s          ! bathymetry file\n'%par)
        f.write('%(deadzone)g       ! deadzone\n'%par)
        f.write('%(optmethod)d      ! optimizaiton method\n'%par)
        f.write('%(convg_c)g        ! convergence criterion\n'%par)
        f.write('%(convg_cm)g       ! convergence criterion model\n'%par)
        f.write('%(max_iter)d       ! max number of nonlinear iterations\n'%par)
        f.write('%(mem_lbfgs)d      ! memory parameter for L-BFGS \n'%par)
        f.write('%(max_cg_iter)d    ! max number of inner CG iteratoins for truncated Newton\n'%par)
        f.write('%(precon_threshold)g     ! threshold parametr for preconditioner\n'%par)
        f.write('%(optdebug)d    !debug option for the optimization routines\n'%par)
        f.write(' '.join(str('%g')%i for i in par['lamTik']))
        f.write('   ! lambda for Tikhonov regularization\n')
        f.write(' '.join(str('%g')%i for i in par['wlamxz']))
        f.write('   ! lambda_x, lambda_z\n')
        f.write('%(preg)g      !prior informaiotn regularization weights\n'%par)
        f.write('%(bounds)d    !bound constraints\n'%par)
        if par['invpars'][0] == 1:
            f.write('%(ubound)g    !upper bound\n'%par)
            f.write('%(lbound)g    !lower bound\n'%par)
        else:
            f.write(' '.join(str('%g')%i for i in par['ubound']))
            f.write('    !upper bound\n')
            f.write(' '.join(str('%g')%i for i in par['lbound']))
            f.write('    !lower bound\n')
        f.write('%(tol_bound)g      !tolerance for bound constraints\n'%par)
        f.write('%(fdataw)s         !file for data weighting\n'%par)
        f.write('%(ndataw)d %(ddataw)g    !number of sample and space step of the weighting file\n'%par)


def runt2d(shcmd, conn):
    tag=1
    pfwi = Popen(shcmd, close_fds=True)
    while tag==1:
        tag = conn.recv()
    if Popen.poll(pfwi) is None:
        time.sleep(5)
        Popen.kill(pfwi)


def checkt2d(fsize, conn):
    while True:
        if os.path.isfile('param_vp_final'): 
            if os.path.getsize('param_vp_final') == fsize:
                conn.send(0)
                break
            else:
                print('writing param_vp_final, wait 5 sec')
                time.sleep(5)
        else:
            print('toy2dac is running, wait 5 sec')
            time.sleep(5)


def fwi1rec(par, imod, fdata, irec, shcmd, flim, fsize=127624):
    """

    :param par:
    :param imod:
    :param fdata:
    :param irec: label of rec array, 'r' or 'l'
    :param shcmd:
    :param flim: [freq_min, freq_max, dfreq]
    :return:
    """
    t2dac_in_inv(par, 'acq'+irec)
    fwi_in(par, fdata + irec)
    if os.path.isfile('param_vp_final'):
        os.remove('param_vp_final')
    t1 = datetime.now()
    conn_run, conn_check = multiprocessing.Pipe()
    p1 = multiprocessing.Process(target=runt2d, args=(shcmd,conn_run))
    p2 = multiprocessing.Process(target=checkt2d, args=(fsize, conn_check))
    p2.start()
    p1.start()
    p2.join()
    p1.join()
    t2 = datetime.now()
    dt0 = t2 - t1
    dt = dt0.seconds + dt0.microseconds / 1e6
    print('mod'+str(imod), ' rec '+irec, 'running time:{:.3f}'.format(dt))
    savelist = ['param_vp_final', 'gradient']
    renamel = ['inv_vp', 'inv_grad']
    for j in range(len(savelist)):
        fwi_out = os.path.join(par['inv_path'], savelist[j])
        if os.path.isfile(fwi_out):
            os.rename(fwi_out,
                      renamel[j] + str(imod)+ irec +
                      str(flim[0])+'_'+str(flim[1])+'_'+str(flim[2])+'Hz')
    for i in ['r', 'l']:
        tempd = os.path.join(par['inv_path'], fdata+i)
        if os.path.isfile(tempd):
            os.remove(tempd)


def fwi1mod(par, imod, irec, freqlist, shcmd, freqmng='freq_management',
            nrec1=105, f0=2, df=1, fsize=127624):
    fname = os.path.join(par['data_path'], par['data_pre'] + str(imod))
    fshape = [par['nfreq'], par['nshots'], par['nrec']]
    fdata = 'dataw' + str(imod)
    tempd = os.path.join(par['inv_path'], fdata)
    datasep(fname, freqlist, fshape, tempd, freqmng, nrec1, f0, df)
    fmax = max(freqlist)
    fmin = min(freqlist)
    dfinv = (fmax - fmin) / (len(freqlist)-1.)
    flim = [fmin, fmax, dfinv]
    if irec =='r' or irec =='l':
        fwi1rec(par, imod, fdata, irec, shcmd, flim, fsize)
    else:
        fwi1rec(par, imod, fdata, 'r', shcmd, flim, fsize)
        fwi1rec(par, imod, fdata, 'l', shcmd, flim, fsize)


def fwi1modx(par, imod, freqlist, shcmd):
    """

    :param par:
    :param imod:
    :param fdata:
    :param irec: label of rec array, 'r' or 'l'
    :param shcmd:
    :param flim: [freq_min, freq_max, dfreq]
    :return:
    """
    # select data
    fname = os.path.join(par['path_data'], par['data_pre'] + str(imod))
    fshape = [par['nfreq'], par['nshots'], par['nrec']]
    data = np.fromfile(fname, dtype=np.complex64)
    data = np.reshape(data, fshape)
    fidx = ((freqlist - par['freq0']) / par['freq_step']).astype(np.int32)
    dataw1 = data[fidx, :, :]
    fout = par['data_inv']
    dataw1.tofile(fout)
    nfreq = len(freqlist)
    with open(par['ffreq_man'], 'w') as f:
        f.write('%d\n' % nfreq)
        freqlist.tofile(f, sep=' ', format='%g')
        f.write('\n')
    if os.path.isfile('param_vp_final'):
        os.remove('param_vp_final')
    t1 = datetime.now()
    conn_run, conn_check = multiprocessing.Pipe()
    p1 = multiprocessing.Process(target=runt2d, args=(shcmd,conn_run))
    p2 = multiprocessing.Process(target=checkt2d, args=(par['inv_fsize'], conn_check))
    p2.start()
    p1.start()
    p2.join()
    p1.join()
    t2 = datetime.now()
    dt0 = t2 - t1
    dt = dt0.seconds + dt0.microseconds / 1e6
    print('mod {:d}, running time:{:.3f}'.format(imod, dt))
    # rename file and clean up
    savelist = ['param_vp_final', 'gradient']
    renamel = ['inv_vp', 'inv_grad']
    fmin = freqlist[0]
    fmax = freqlist[-1]
    dfinv = (fmax - fmin) / (len(freqlist)-1.)
    for j in range(len(savelist)):
        fwi_out = os.path.join(par['inv_path'], savelist[j])
        if os.path.isfile(fwi_out):
            os.rename(fwi_out,'{:s}{:d}_{:g}_{:g}_{:g}Hz'.format(
                renamel[j], imod, fmin, fmax, dfinv))
    tempd = os.path.join(par['inv_path'], fout)
    if os.path.isfile(tempd):
        os.remove(tempd)


def fwi1modx1(par, imod, freqlist, shcmd):
    """

    :param par:
    :param imod:
    :param fdata:
    :param irec: label of rec array, 'r' or 'l'
    :param shcmd:
    :param flim: [freq_min, freq_max, dfreq]
    :return:
    """
    # select data
    fname = os.path.join(par['path_data'], par['data_pre'].format(imod))
    fshape = [par['nfreq'], par['nshots'], par['nrec']]
    data = np.fromfile(fname, dtype=np.complex64)
    data = np.reshape(data, fshape)
    fidx = ((freqlist - par['freq0']) / par['freq_step']).astype(np.int32)
    dataw1 = data[fidx, :, :]
    fout = par['data_inv']
    dataw1.tofile(fout)
    nfreq = len(freqlist)
    with open(par['ffreq_man'], 'w') as f:
        f.write('%d\n' % nfreq)
        freqlist.tofile(f, sep=' ', format='%g')
        f.write('\n')
    if os.path.isfile('param_vp_final'):
        os.remove('param_vp_final')
    t1 = datetime.now()
    conn_run, conn_check = multiprocessing.Pipe()
    p1 = multiprocessing.Process(target=runt2d, args=(shcmd,conn_run))
    p2 = multiprocessing.Process(target=checkt2d, args=(par['inv_fsize'], conn_check))
    p2.start()
    p1.start()
    p2.join()
    p1.join()
    t2 = datetime.now()
    dt0 = t2 - t1
    dt = dt0.seconds + dt0.microseconds / 1e6
    print('mod {:d}, running time:{:.3f}'.format(imod, dt))
    # rename file and clean up
    savelist = ['param_vp_final', 'gradient']
    renamel = ['inv_vp', 'inv_grad']
    fmin = freqlist[0]
    fmax = freqlist[-1]
    dfinv = (fmax - fmin) / (len(freqlist)-1.)
    for j in range(len(savelist)):
        fwi_out = os.path.join(par['inv_path'], savelist[j])
        if os.path.isfile(fwi_out):
            os.rename(fwi_out,'{:s}{:d}_{:g}_{:g}_{:g}Hz'.format(
                renamel[j], imod, fmin, fmax, dfinv))
    tempd = os.path.join(par['inv_path'], fout)
    if os.path.isfile(tempd):
        os.remove(tempd)


def checkt2dfd(fsize, conn):
    while True:
        if os.path.isfile('data_modeling'):
            if os.path.getsize('data_modeling') == fsize:
                conn.send(0)
                break
            else:
                print('writing data_modeling, wait 1 sec')
                time.sleep(1)
        else:
            time.sleep(1)


def t2dfd(par, imod, shcmd):
    par['fvpinit'] = par['mod_pre'] + str(imod)+'.bin'
    fd_in_iso_inv(par)
    srcfvp = os.path.join(par['mod_true_path'], par['fvpinit'])
    tgtfvp = os.path.join(par['inv_path'], par['fvpinit'])
    copyfile(srcfvp, tgtfvp)
    if os.path.isfile('data_modeling'):
        os.remove('data_modeling')
    t1 = datetime.now()
    conn_run, conn_check = multiprocessing.Pipe()
    p1 = multiprocessing.Process(target=runt2d, args=(shcmd,conn_run))
    p2 = multiprocessing.Process(target=checkt2dfd, args=(par['data_size'], conn_check))
    p2.start()
    p1.start()
    p2.join()
    p1.join()
    t2 = datetime.now()
    dt0 = t2 - t1
    dt = dt0.seconds + dt0.microseconds / 1e6
    print('mod' + str(imod), 'running time:{:.3f}'.format(dt))
    if os.path.isfile('data_modeling'):
        os.rename('data_modeling', par['data_pre']+str(imod))
    os.remove(tgtfvp)


class InitMod:
    def __init__(self, datadir, basename, shape, dtype=np.float32):
        self.datadir = datadir
        self.basename = basename
        self.shape = shape
        self.dtype = dtype

    def loadata(self, imod):
        fname = self.basename.format(imod)
        with open(os.path.join(self.datadir, fname), 'rb') as f:
            data = np.fromfile(f, dtype=self.dtype).reshape(self.shape)
        return data

    def smooth(self, imod, ksize, rep, sigma=None, show_fig=False):
        """
        Smooth an image using RectBlur or GaussBlur
        :param imod: int, mod id
        :param ksize: tuple, int, kernel size
        :param rep: int, repeatition
        :param sigma: float
        :return:
        """
        img0 = self.loadata(imod)
        ksize = tuple(ksize)
        for i in range(rep):
            if sigma is None:
                img = cv.blur(img0, ksize)
            else:
                img = cv.GaussianBlur(img0, ksize, sigma, sigma)
        img = img.astype(self.dtype)
        if show_fig:
            self.show_fig(img0.T, img.T)
        return img

    def show_fig(self, modtrue, modsmooth, trans=True, same_scale=True):
        if trans:
            modtrue = modtrue.T
            modsmooth = modsmooth.T
        vmin = modtrue.min()
        vmax = modtrue.max()
        fig, ax = plt.subplots(1 ,2)
        if same_scale:
            ax[0].imshow(modtrue, cmap='jet', vmin=vmin, vmax=vmax)
            ax[1].imshow(modsmooth, cmap='jet', vmin=vmin, vmax=vmax)
        else:
            ax[0].imshow(modtrue, cmap='jet')
            ax[1].imshow(modsmooth, cmap='jet')
        plt.show()