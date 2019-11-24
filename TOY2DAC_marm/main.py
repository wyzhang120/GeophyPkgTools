import os
import fwiprep
from utils_marmousi_cw import *
from util_model_building import mod2d
from shutil import copyfile


segydir = '/project/stewart/wzhang/TOY2DAC/crosswell_toy2dac/FWI/Mamousi/model_full'
fvp = 'MODEL_P-WAVE_VELOCITY_1.25m.segy' # unit = m/s
frho = 'MODEL_DENSITY_1.25m.segy' # unit = g/cc
# boundary box, unit = m
x0 = 9100
x1 = 9600
z0 = 1600
z1 = 2600
dxSgy = 1.25

# acquisition params
fc = 120 # fmax = 2.76 * fc
acqOffset = 12.5 # unit = m, dx = 1.25
acqDz = 2.5
srcpar = {'x0': acqOffset, 'zmin': acqOffset, 'zmax': z1 - z0 - acqOffset, 'dz': acqDz}
recpar = {'x0': x1 - x0 - acqOffset, 'zmin': acqOffset, 'zmax': z1 - z0 - acqOffset, 'dz': acqDz}

# toy2dac params
t2dpar = {
    # TOY2DAC
    'fvpinit': 'vp_init_mamousi', 'fvpinit0': 'vp_init_mamousi',
    'sigma_init_mod':20, 'repeat':5, # smmothing to get init model
    'toymode': 1, # 0 = forward, 1 = fwi
    # inversion window
    'inv_z_offset':100, #  vertical src-rec offset
    'inv_freq0':2, 'inv_freq1':31, # min, max freq
    # directy where $toy2dacbin will be run
    'path_fd': '/project/stewart/wzhang/TOY2DAC/crosswell_toy2dac/FWI/Mamousi/fdrun', # toymode = 0
    'path_fwi': '/project/stewart/wzhang/TOY2DAC/crosswell_toy2dac/FWI/Mamousi/fwirun', # toymod = 1
    # fvp,fqp,frho
    'sigma_true_mod':3, # smoothing to get model for forward modeling
    'fvp': 'vp_mamousi', 'fqp': 'qp_mamousi', 'frho': 'rho_mamousi',
    'qp_init': 'qp_mamousi', 'rho_init': 'rho_mamousi',
    'qp': 1000.,
    # acquisitin file name
    'facq': 'acq',

    # freq_management file
    'ffreq_man': 'freq_management',
    'freq0': 2, 'freq-1': 301, 'freq_step': 1,


    # toy2dac_input file
    'f2dac': 'toy2dac_input',
    # mode=0 frequncy modeling; 1 FWI; not working 2 RTM, 3 MVA, 4 time domain modeling
    'iso': 1,
    # iso=1 fdfd iso; 2 fdfd aniso

    # fdfd_input file
    'ffd': 'fdfd_input',
    'pml_coef': 90.,
    # pml tuning with the value of the pml amplitude (90 is a good pragmatical value)
    'npml': 50,
    'Hicks_interp': 1,  # Hicks interpolation (0 no; 1 yes)
    # Interpolation from Hicks when sources and/or receivers are not on grid points
    'free_surf': 0,  # free surface (0 no; 1 yes)
    'srctype': 0,  # source type (0 explosion; 1 vertical force)
    'rcvtype': 0,  # receiver type (0 hydrophone; 1 vertical geophone)
    'slaplace': 0.,  # Laplace constant, i.e., imaginary part of frequency used to
    # exponentially damp the seismic wavefile with time

    # bathymetry file (binary)
    'fbath': 'bathym',
    'seafloor': 0.,

    # data weighting
    'fdataw': 'data_weight_file',  # file for data weighting
    'ndataw': 2,  # number of sample the weighting file
    'ddataw': 10.,  # space step (src-rec distance) of the weighting file
    'dataw': 1.,  # uniform weight


    # fwi_input file
    'ffwi': 'fwi_input',
    'fwdata': 'data_modeling',  # name of observed data
    'family': 1,  # parametrization family choice, only choice 1 available currently
    'invpars': [1, 1],  # number of parameters for invsion,
    # followed by code(s) of target parameter(s)
    # 1 vp; 2 rho; 3 qp; 4 eps; 5 delta; 11 log(vp);12 log(rho); 31 log(vp/vp0)
    # 32 log(rho/rho0); 33 log(Qp/Qp0); 34 log(1+epsilon); 35 log(1+delta)
    'lamTik': [1e-2],  # lamda for smoothing/Tikhonov regularization;
    # length of the list should be the same as # of inverted paramters
    # If negative value, that activates the smoothing of the gradient
    # If positive value, this is the absolute regularization weights per parameter
    # class for first order derivatives Tikhonov regularization.
    'wlamxz': [.5, .5],  # lamda_x, lamda_z (directional weight for smooting/Tik reg)
    'src_est': 0,  # src wavelet estimation
    # 0 no; 1 one estimation for all gathers; 2 one per gather
    'deadzone': 0.,
    # a distance, considered bellow the bathymetry depth, where the properties
    # of the media are not involved during inversion,
    # but the properties can changed due to regularization
    'optmethod': 6,
    # 1 steepest descent; 2 preconditioned steepest descent; 3 nonlinear conjugate gradient
    # 4 preconditioned nonlinear conjugate gradient; 5 L-BFGS; 6 preconditioned L-BFGS
    # 7 truncated Gauss-Newton; 8 truncated Newton; 9 precision truncated Gauss-Newton
    # 10 preconditioned truncated Newton
    # pseudo Hessian is used for preconditioning; it's only adapted to surface acquistion
    'convg_c': 1e-4,  # convergence criterion
    # Iteration stops as soon as f(m)/f(m0) becomes smaller tha convg_c
    'convg_cm': 1e-4,  # convergence criterion model update
    'max_iter': 10,  # max number of iterations
    'mem_lbfgs': 20,  # memory parameter for L-BFGS
    'max_cg_iter': 3,  # max inner CG iterations for truncated Newton
    'precon_threshold': 1e-2,  # threshold parameter for the preconditioner
    # Samller value has stronger impact on ivnersion. Too small yields instabilities.
    'optdebug': 1,  # 1 addtionaly pritings in output file give info on linesearch process
    'preg': 0.,  # prior information regularization weights (not implemented)
    'bounds': 1,  # bound constraints 1 yes; 0 no.
    'tol_bound': 0.,  # tolerance for bound constraints in the unit of the reconstructed paramter

    # mumps
    'fmumps': 'mumps_input',
    'icntl_7': 7,
    # icntl_7= choice of ordering (default 7->automatic choice)
    # By default, we recommend to use the 7 value that most of the time
    # chooses the METIS algorithm for our matrices
    'icntl_14': 60,
    # icntl_14=percentage of increasing of estimated workspace for factorization
    # (default=20)
    # This option allows that MUMPS allocates a bigger table that estimated
    # during analysis (roughly: size = (1+icntl_14=100)*estimated size)
    'icntl_23': 9000,
    # icntl_23=MEMORY USED PER PROCESSSOR
    # The icntl 23 parameter gives the allowed memory in Mb per MPI process during
    # factorization (used since MUMPS version 4.9.X). This flag is really important
    # as it determine the size of the table allowed on each MPI process to store the LU
    # factors.
    'keep_84': 16,
    # keep_84 = blocking factor for Multiple RHS
    # The keep 84 parameter allows to choose the number of shots (right hand side terms
    # of the linear system) solved simultaneously with MUMPS.  We recommend to use a
    # maximum value of 16, which often works fine. Smaller values of 8 or 4 can also be used.
    }


vp = crop_mamousi(segydir, fvp, x0, x1, z0, z1)
rho = crop_mamousi(segydir, frho, x0, x1, z0, z1) * 1000
t2dpar['ubound'] = vp.max()
t2dpar['lbound'] = vp.min()
mod2dDict0 = {'vp': vp, 'rho': rho}

# acquisition obj and mod2d obj
acqObj = AcqCw(srcpar, recpar)
acqDict0 = acqObj.getAcqDict()
modObj = mod2d(mod2dDict0, acqDict0, dxSgy, dxSgy)
modObj.fdParams(fc, 1, '8')
nx, nz = modObj.vp.shape
t2dpar['nz'] = nz
t2dpar['nx'] = nx
t2dpar['dx'] = modObj.dx
t2dpar['dz'] = modObj.dz
nfreq = int((t2dpar['freq-1'] - t2dpar['freq0']) / t2dpar['freq_step'])
# toy2dac input: acq file
pars = {'nshots': len(modObj.zsrc), 'xsrc': srcpar['x0'], 'osV': srcpar['x0'],
        'inv_path': t2dpar['path_fd']}
parr = {'nrV': len(modObj.zrec), 'xrec': recpar['x0'], 'orV': recpar['zmin'],
        'facq': t2dpar['facq']}
if t2dpar['toymode'] ==0:
    t2dpar['inv_path'] = t2dpar['path_fd']
    fwiprep.acq(pars, parr, acqDz, acqDz)
    # toy2dac input files
    fwiprep.write_vp_rho_qp(t2dpar, modObj.vp, modObj.rho) # vp, rho, qp
    fwiprep.t2dac_in_inv(t2dpar, t2dpar['facq'])  #toy2dac_input
    fwiprep.mumps_in(t2dpar) # mumps_input
    fwiprep.bathym(t2dpar) # bathymetry file
    fwiprep.data_weight_file(t2dpar) # data weight file
    fwiprep.fd_in_iso(t2dpar) # fdfd_input using true model
    fwiprep.freqm(t2dpar) # freq_management
    fwiprep.fwi_in(t2dpar, t2dpar['fwdata']) # fwi_input
else:
    t2dpar['inv_path'] = t2dpar['path_fwi']
    copyfile(os.path.join(t2dpar['path_fd'], t2dpar['facq']),
             os.path.join(t2dpar['inv_path'], t2dpar['facq']))
    copyfile(os.path.join(t2dpar['path_fd'], t2dpar['fwdata']),
             os.path.join(t2dpar['inv_path'], t2dpar['fwdata']))
    invfreqlist = np.arange(t2dpar['inv_freq0'], t2dpar['inv_freq1'], t2dpar['freq_step'])
    # invfreqlist = np.array([2, 30, 45, 67, 100, 150, 225])
    fshape = (nfreq, pars['nshots'], parr['nrV'])
    fout = 'data_fwi'
    t2dpar['ndataw'] = fwiprep.data_weight_voffset(t2dpar, srcpar, recpar) # data weight file
    fwiprep.mumps_in(t2dpar)  # mumps_input
    fwiprep.data_window(os.path.join(t2dpar['inv_path'], t2dpar['fwdata']),
                        invfreqlist, fshape, os.path.join(t2dpar['inv_path'], fout),
                        os.path.join(t2dpar['inv_path'],t2dpar['ffreq_man']),
                        t2dpar['freq0'], t2dpar['freq-1'])
    fwiprep.t2dac_in_inv(t2dpar, t2dpar['facq'])  # toy2dac_input
    if not os.path.exists(os.path.join(t2dpar['inv_path'], t2dpar['fvpinit0'])):
        fwiprep.write_init_mod(modObj.vp, t2dpar) # initial vp file
    fwiprep.fd_in_iso_inv(t2dpar)  # fdfd_input using fvpinit
    fwiprep.fwi_in(t2dpar, fout)  # fwi_input


# freq_arr = np.arange(par['freq0'], par['freq-1'], par['freq_step'])
# par['freqs'] = freq_arr.tolist()
# par['nfreq'] = len(par['freqs'])
# # padding for displacement from absorbding boundary condition
# par['padx'] = par['padx'] + int(par['padxm'] / par['dx'])
# par['padz'] = par['padz'] + int(par['padzm'] / par['dz'])
#
# parr = {'nrV': par['nrec1'],
#         'xrec': par['xrec1'], 'orV': par['orV1'], 'jrV': par['jrV'],
#         'facq': 'acqr'}
