import os
import numpy as np
import denise_IO.denise_out as deniseIO
from denise_IO.utils_denise import AcqCw2, print_factors, write_mfile
from utils_marmousi_cw import crop_mamousi
from util_model_building import mod2d
from sympy.ntheory import factorint


basedir = '/project/stewart/wzhang/src/DENISE-Black-Edition/par_fdtest'
fmodel = 'CW_fdtest'
fdenise_input = '{:s}.inp'.format(fmodel)
fsrc = '{:s}_src.txt'.format(fmodel)
frec = '{:s}_rec'.format(fmodel)
fd_order = 8
# Mamousi
segydir = '/project/stewart/wzhang/TOY2DAC/crosswell_toy2dac/FWI/Mamousi/model_full'
fvp = 'MODEL_P-WAVE_VELOCITY_1.25m.segy' # unit = m/s
fvs = 'MODEL_S-WAVE_VELOCITY_1.25m.segy'
frho = 'MODEL_DENSITY_1.25m.segy' # unit = g/cc
dxSgy = 1.25
npml = 20
dpml = npml * dxSgy
x0 = 9100 - dpml
x1 = 9600 + dpml - dxSgy
z0 = 1600 - dpml
z1 = 2600 + dpml - dxSgy

fc = 80 # fmax = 2.76 * fc
srctype = 1 # 1=explosive, 2=point force in x, 3=point force in z, 4=custom directive force
acqOffset = 25 + dpml # unit = m, dx = 1.25
acqDz = 100
tmax = 2.
vp = crop_mamousi(segydir, fvp, x0, x1, z0, z1)
vs = crop_mamousi(segydir, fvs, x0, x1, z0, z1)
rho = crop_mamousi(segydir, frho, x0, x1, z0, z1) * 1000
srcpar = {'x0': acqOffset, 'zmin': acqOffset, 'zmax': z1 - z0 - acqOffset, 'dz': acqDz,
          'fname': os.path.join('source', fsrc), 'fc': fc, 'srctype': srctype, 'amp': 1.}
recpar = {'x0': x1 - x0 - acqOffset, 'zmin': acqOffset, 'zmax': z1 - z0 - acqOffset, 'dz': acqDz,
          'fname': os.path.join('receiver', frec)}
acqObj = AcqCw2(srcpar, recpar)

mod2dDict0 = {'vp': vp, 'rho': rho}
acqDict0 = acqObj.acqdict
modObj = mod2d(mod2dDict0, acqDict0, dxSgy, dxSgy)
modObj.fdParams(fc, tmax, '{:d}'.format(fd_order))

para={
    'filename': os.path.join(basedir, 'CW_fdtest.inp'),
    'descr': fmodel,
    'MODE': 0, # forward_modelling_only=0;FWI=1;RTM=2
    'PHYSICS': 2, # 2D-PSV=1;2D-AC=2;2D-VTI=3;2D-TTI=4;2D-SH=5
    # domain decomposition and 2D grid
    'NPROCX': 4, 'NPROCY': 7, 'NX': 100, 'NY': 100, 'DH': 1,
    # time stepping
    'TIME': tmax, 'DT': 1e-3,
    # FD params
    'FD_ORDER': 8,
    'max_relative_error': 0, # Taylor (max_relative_error=0) and Holberg (max_relative_error=1-4)

    # source
    'QUELLART': 1, # ricker=1 (delayed 1.5 / fc);fumue=2;from_SOURCE_FILE=3;SIN**3=4;Gaussian_deriv=5;Spike=6;Klauder=7
    'SOURCE_FILE': os.path.join('./source', fsrc),
    'SIGNAL_FILE': './wavelet/wavelet_cw',
    'TS': 8, # duration_of_Klauder_wavelet_(in_seconds)
    'SRCTYPE': srctype, # 1=explosive, 2=point force in x, 3=point force in z, 4=custom directive force
    'RUN_MULTIPLE_SHOTS': 1, # multiple shots one by one
    'FC_SPIKE_1': -5, # corner_frequency_of_highpass_filtered_spike
    'FC_SPIKE_2': 15, # orner_frequency_of_lowpass_filtered_spike
    'ORDER_SPIKE': 5, # order_of_Butterworth_filter
    'WRITE_STF': 0, # write_source_wavelet
    # model
    'READMOD': 1, # read_model_parameters_from_MFILE(yes=1)
    'MFILE': os.path.join('./model', fmodel),
    'WRITEMOD': 0,
    # boundary conditions
    'FREE_SURF': 0,
    'FW': npml, # width_of_absorbing_frame_(in_gridpoints)
    'DAMPING': 3000, # Damping_velocity_in_CPML_(in_m/s)
    'FPML': fc, # Frequency_within_the_PML_(Hz)
    'npower': 2.0, 'k_max_PML': 1.0,
    # Q approximation
    'L': 0, # Number_of_relaxation_mechanisms
    'FL': 20000, # L_Relaxation_frequencies
    # snapshots
    'SNAP': 1, # output_of_snapshots_(SNAP)(yes>0)
    'SNAP_SHOT': 1, # write_snapshots_for_shot_no_(SNAP_SHOT)
    'TSNAP1': 0.05, # first_snapshot_(in_sec)
    'TSNAP2': 1, # last_snapshot_(in_sec)
    'TSNAPINC': 0.05, # increment_(in_sec)
    'IDX': 1, # increment_x-direction
    'IDY': 1, # increment_y-direction
    'SNAP_FORMAT': 3, # data-format 2=ASCII, 3=BINARY
    'SNAP_FILE': './snap/{:s}_waveform_forward'.format(fmodel), # basic_filename
    # receiver input
    'READREC': 1, # read_receiver_positions_from_file, 1=single_file, 2=multiple_files
    'REC_FILE': os.path.join('./receiver', frec),
    # towed streamer
    'N_STREAMER': 0, # The_first_(N_STREAMER)_receivers_in_REC_FILE_belong_to_streamer
    'REC_INCR_X': 80, 'REC_INCR_Y': 0, # Cable_increment_per_shot
    # seismogram
    'SEISMO': 2, # output_of_seismograms, 0: no seismograms; 1: particle-velocities;
                 # 2: pressure (hydrophones); 3: curl and div; 4: everything
    'NDT': 1, # samplingrate_(in_timesteps!)
    'SEIS_FORMAT': 1, # SU(1);ASCII(2);BINARY(3)
    'SEIS_FILE_VX': os.path.join('./seismo', '{:s}_vx'.format(fmodel)), # filename_for_Vx
    'SEIS_FILE_VY': os.path.join('./seismo', '{:s}_vy'.format(fmodel)), # filename_for_Vy
    'SEIS_FILE_CURL': os.path.join('./seismo', '{:s}_curl'.format(fmodel)), # filename_for_curl
    'SEIS_FILE_DIV': os.path.join('./seismo', '{:s}_div'.format(fmodel)), # filename_for_div
    'SEIS_FILE_P': os.path.join('./seismo', '{:s}_p'.format(fmodel)), # ilename_for_pressure
    # log file
    'LOG_FILE': os.path.join('./log', fmodel), # log-file_for_information_about_progress_of_program
    'LOG': 2, # 0=no log; 1=PE 0 writes this info to stdout; 2=PE 0 also outputs information to LOG_FILE.0

    # FWI
    'ITERMAX': 100, # number_of_TDFWI_iterations
    'JACOBIAN': 'jacobian/jacobian_test', # output_of_gradient
    'DATA_DIR': 'su/MARMOUSI_spike/DENISE_MARMOUSI', # seismograms_of_measured_data
    'TAPER': 0, # cosine_taper_(yes=1/no=0)
    'TAPERLENGTH': 4, # taper_length_(in_rec_numbers)
    'GRADT1': 21, 'GRADT2': 25, 'GRADT3': 490, 'GRADT4': 500, # gradient_taper_geometry
    'INVMAT1': 1, # type_of_material_parameters_to_invert_(Vp,Vs,rho=1; Zp,Zs,rho=2; lam,mu,rho=3)
    'QUELLTYPB': 1, # adjoint_source_type_(x-y_components=1, y_comp=2,
                    # x_comp=3, p_comp=4, x-p_comp=5, y-p_comp=6, x-y-p_comp=7)
    'TESTSHOT_START': 25, 'TESTSHOT_END': 75, 'TESTSHOT_INCR': 10,# testshots_for_step_length_estimation
    # gradient taper geometry
    'SWS_TAPER_GRAD_VERT': 0, # apply_vertical_taper_(yes=1)
    'SWS_TAPER_GRAD_HOR': 0, # apply_horizontal_taper
    'EXP_TAPER_GRAD_HOR': 2.0, # exponent_of_depth_scaling_for_preconditioning
    'SWS_TAPER_GRAD_SOURCES': 0, # apply_cylindrical_taper_(yes=1)
    'SWS_TAPER_CIRCULAR_PER_SHOT': 0, # apply_cylindrical_taper_per_shot
    'SRTSHAPE': 1, # damping shape (1=error_function,2=log_function)
    'SRTRADIUS': 5., # radius_in_m, minimum for SRTRADIUS is 5x5 gridpoints
    'FILTSIZE': 1, # filtsize_in_gridpoints
    'SWS_TAPER_FILE': 0, # read_taper_from_file_(yes=1)
    # output of ivnerted models
    'INV_MOD_OUT': 1, # write_inverted_model_after_each_iteration_(yes=1)
    'INV_MODELFILE':'model/modelTest', # output_of_models
    # upper and lower limits of model params
    'VPLOWERLIM': 3000, 'VPUPPERLIM': 4500, # lower/upper_limit_for_vp/lambda
    'VSLOWERLIM': 1500, 'VSUPPERLIM': 2250, # lower/upper_limit_for_vs/mu
    'RHOLOWERLIM': 2000, 'RHOUPPERLIM': 2600, # lower/upper_limit_for_rho
    'QSLOWERLIM': 10, 'QSUPPERLIM': 100, # ower/upper_limit_for_Qs
    # optimization method
    'GRAD_METHOD': 2, # gradient_method_(PCG=1/LBFGS=2)
    'PCG_BETA': 2, # preconditioned conjugate gradeint
                   # Fletcher_Reeves=1, Polak_Ribiere=2, Hestenes_Stiefel=3, Dai_Yuan=4
    'NLBFGS': 20, # save_(NLBFGS)_updates_during_LBFGS_optimization
    # smoothing models
    'MODEL_FILTER': 0, # apply_spatial_filtering_(1=yes)
    'FILT_SIZE': 5, # filter_length_in_gridpoints
    # Reduce size of inversion grid
    'DTINV': 3, # use_only_every_DTINV_time_sample_for_gradient_calculation
    # step length estimation
    'EPS_SCALE': 0.01, # maximum_model_change_of_maximum_model_value
    'STEPMAX': 6, # maximum_number_of_attemps_to_find_a_step_length
    'SCALEFAC': 2.0,
    # trace killing
    'TRKILL': 0, # apply_trace_killing_(yes=1)
    'TRKILL_FILE': './trace_kill/trace_kill.dat', #
    # time damping
    'PICKS_FILE': './picked_times/picks_', # files_with_picked_times
    # FWI log
    'MISFIT_LOG_FILE': 'LOG_TEST.dat', # log_file_for_misfit_evolution
    'MIN_ITER': 0, # minimum number of iteration per frequency
    # Definition of smoothing the Jacobians with 2D-Gaussian
    'GRAD_FILTER': 0, # apply_spatial_filtering_(yes=1)
    'FILT_SIZE_GRAD': 10, # filter_length_in_gridpoints
    # FWT double-difference time-lapse mode
    'TIMELAPSE': 0, # activate_time_lapse_mode_(yes=1); if TIMELAPSE == 1,
                    # DATA_DIR should be the directory containing the data differences
    'DATA_DIR_T0': 'su/CAES_spike_time_0/DENISE_CAES', # seismograms_of_synthetic_data_at_t0_()

    # RTM
    'RTMOD': 0, # apply_reverse_time_modelling_(yes=1)
    'RTM_SHOT': 0, # output_of_RTM_result_for_each_shot_(yes=1)
    # gravity modeling/inversion
    'GRAVITY': 0, # 0 = no gravity modeling, 1 = active_gravity_modelling_, 2=inversion
}

para['DH'] = modObj.dx
para['DT'] = modObj.dt
para['TIME'] = modObj.dt * modObj.nt
nx, nz = modObj.vp.shape
vptest = np.ones((nx, nz), dtype=np.float32) * 3000
vstest = np.ones((nx, nz), dtype=np.float32) * 3000
rhotest = np.ones((nx, nz), dtype=np.float32) * 2000
deniseIO.calc_max_freq(vptest, vstest, para)
deniseIO.check_stability(vptest, vstest, para)
para['FW'] = int(dpml / para['DH'])
para['DAMPING'] = 3000
print_factors(nx, nz)
para['NPROCX'] = 4
para['NPROCY'] = 7
para['NX'] = nx
para['NY'] = nz
deniseIO.check_domain_decomp(para)
deniseIO.write_denise_para(para)
acqObj.write_acq(basedir)
dict_mfile = {'vp': vptest, 'vs': vstest, 'rho': rhotest}
write_mfile(para['MFILE'], dict_mfile, basedir)
