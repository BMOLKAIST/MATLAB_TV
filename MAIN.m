%% 1. Set folder
clc;clear;
use_GPU = true;
cd0 = matlab.desktop.editor.getActiveFilename;
dash = cd0(strfind(cd0,'MAIN')-1);
cd0 = cd0(1:strfind(cd0,'MAIN')-2);
addpath(genpath(cd0));
cds.data = [cd0 dash 'Data'];

% 1. ODT raw data
cds.bg_file = [cds.data dash '1. ODT raw data' dash 'SiO2_1_bg.tif'];
cds.sp_file = [cds.data dash '1. ODT raw data' dash 'SiO2_1_sp.tif'];
% 2. PEPSI-processed data
cds.PEPSI = [cds.data dash '2. PepsiODT - processed data' dash 'PEPSI_data.mat'];
% 3. FL raw data
cds.FL = [cds.data dash '3. Fluorescence' dash 'FL_data.mat'];

%% 1. TV to ODT
MULTI_GPU=false;
params=BASIC_OPTICAL_PARAMETER();
params.NA=1.16;
params.RI_bg=1.3355;
params.wavelength=0.532;
params.resolution=[1 1 1]*params.wavelength/4/params.NA;
params.vector_simulation=false;true;
params.size=[0 0 71]; 
params.use_GPU = use_GPU;

%2 illumination parameters
field_retrieval_params=FIELD_EXPERIMENTAL_RETRIEVAL.get_default_parameters(params);
field_retrieval_params.resolution_image=[1 1]*(5.5/100);
field_retrieval_params.conjugate_field=true;
field_retrieval_params.use_abbe_correction=true;

% 1. Aberration correction
field_retrieval=FIELD_EXPERIMENTAL_RETRIEVAL(field_retrieval_params);

% Aberration correction data
[input_field,field_trans,params]=field_retrieval.get_fields(cds.bg_file,cds.sp_file);
% figure;orthosliceViewer(squeeze(abs(field_trans(:,:,:)./input_field(:,:,:))),'displayrange',[0 2]); colormap gray; title('Amplitude')
% figure;orthosliceViewer(squeeze(angle(field_trans(:,:,:)./input_field(:,:,:)))); colormap jet; title('Phase')

% RI - rytov
rytov_params=BACKWARD_SOLVER_RYTOV.get_default_parameters(params);
rytov_params.use_non_negativity=false;
rytov_solver=BACKWARD_SOLVER_RYTOV(rytov_params);
[RI_rytov, ORytov]=((rytov_solver.solve(input_field,field_trans)));
RI_rytov = real(RI_rytov);
mask = ORytov ~=0;
% figure;orthosliceViewer(real(RI_rytov)); title('Rytov')

% RI - TV
backward_single_params=BACKWARD_SOLVER_SINGLE.get_default_parameters(params);
init_backward_single_params=BACKWARD_SOLVER_RYTOV.get_default_parameters(params);
init_solver_backward_single=BACKWARD_SOLVER_RYTOV(init_backward_single_params);
backward_single_params.init_solver=rytov_solver;
backward_single_params.verbose = false;

% Main parameters
backward_single_params.use_non_negativity = true;
backward_single_params.itter_max=30;
backward_single_params.inner_itt=100;
backward_single_params.nmax = inf;
backward_single_params.nmin = params.RI_bg;
backward_single_params.step = 1e-3;
backward_single_params.tv_param = 2.5e-2;
backward_single_params.use_GPU = params.use_GPU;
single_solver=BACKWARD_SOLVER_SINGLE(backward_single_params);
t1 = clock;
RI_TV=single(real(single_solver.solve(input_field,field_trans,RI_rytov,mask)));
elapse_time = etime(clock,t1);
figure;orthosliceViewer(real(cat(2, RI_rytov,RI_TV))); title('Rytov only vs TV')

%% 2. TV to PEPSI
% Load data
load(cds.PEPSI)
params2=BASIC_OPTICAL_PARAMETER();
params2.NA=0.75;%NA_obj;
params2.RI_bg=1.495;%n_m;
params2.wavelength=0.449;%lambda;
params2.resolution=[0.162 0.162 0.73];%[lateral_resolution lateral_resolution axial_resolution];
params2.size=size(RI_PEPSI);
params2.use_GPU = use_GPU;

% Initialize solver
rytov_params=BACKWARD_SOLVER_RYTOV.get_default_parameters(params2);
rytov_params.use_non_negativity=false;
rytov_solver=BACKWARD_SOLVER_RYTOV(rytov_params);
backward_single_params=BACKWARD_SOLVER_SINGLE.get_default_parameters(params2);
init_backward_single_params=BACKWARD_SOLVER_RYTOV.get_default_parameters(params2);
init_solver_backward_single=BACKWARD_SOLVER_RYTOV(init_backward_single_params);
backward_single_params.init_solver=rytov_solver;
backward_single_params.verbose = false;

% Main parameters
backward_single_params.use_non_negativity = true;
backward_single_params.itter_max=30;
backward_single_params.inner_itt=100;
backward_single_params.nmax = inf;
backward_single_params.nmin = params2.RI_bg;
backward_single_params.step = 1e-3;
backward_single_params.tv_param = 5e-3;
backward_single_params.use_GPU = use_GPU;

% Run TV
single_solver=BACKWARD_SOLVER_SINGLE(backward_single_params);
[mask, OTF] = single_solver.mask_theoretical_OTF(RI_PEPSI, params2.NA, params2.NA);
RI_PEPSI_TV=single_solver.solve(0,0,RI_PEPSI,mask);
figure;orthosliceViewer(real(cat(2, RI_PEPSI,RI_PEPSI_TV))); title('Rytov only vs TV')

%% TV to FL
load(cds.FL)
FL_raw = FL_raw - min(FL_raw(:));
params3=BASIC_OPTICAL_PARAMETER();
params3.NA=FL_params.NA;
params3.RI_bg=FL_params.n_m;
params3.lambda = FL_params.lambda;
params3.resolution=FL_params.resolution;
params3.resolution_image=FL_params.resolution;
params3.size=size(FL_raw);
FL_params=FL_SOLVER.get_default_parameters(params3);
FL_params.RL_iterations = 10;
FL_params.itter_max=30;
FL_params.step = 1;
FL_params.tv_param = 1e-3;
FL_params.use_non_negativity = true;
FL_params.inner_itt =100;
FL_params.verbose = false;
fl_solver=FL_SOLVER(FL_params);

I_RL = fl_solver.solve_WF(FL_raw); % Due to low visibility, temporailly WF deconv is used.
I_TV = fl_solver.solve_WF_TV(FL_raw); % Due to low visibility, temporailly WF deconv is used.
figure;orthosliceViewer(real(cat(2, FL_raw,I_RL, I_TV))); 


%% TV with cuda
% Load data
load(cds.PEPSI)
params2=BASIC_OPTICAL_PARAMETER();
params2.NA=0.75;%NA_obj;
params2.RI_bg=1.495;%n_m;
params2.wavelength=0.449;%lambda;
params2.resolution=[0.162 0.162 0.73];%[lateral_resolution lateral_resolution axial_resolution];
params2.size=size(RI_PEPSI);
params2.use_GPU = use_GPU;

pot2RI=@(pot) single(params2.RI_bg*sqrt(1+pot./((2*pi*params2.RI_bg/params2.wavelength).^2)));
RI2pot=@(RI)  single((2*pi*params2.RI_bg/params2.wavelength)^2*(RI.^2/params2.RI_bg^2-1));

% Initialize solver
rytov_params=BACKWARD_SOLVER_RYTOV.get_default_parameters(params2);
rytov_params.use_non_negativity=false;
rytov_solver=BACKWARD_SOLVER_RYTOV(rytov_params);
backward_single_params=BACKWARD_SOLVER_SINGLE.get_default_parameters(params2);
init_backward_single_params=BACKWARD_SOLVER_RYTOV.get_default_parameters(params2);
init_solver_backward_single=BACKWARD_SOLVER_RYTOV(init_backward_single_params);
backward_single_params.init_solver=rytov_solver;
backward_single_params.verbose = false;

% Main parameters
backward_single_params.use_non_negativity = true;
backward_single_params.itter_max=30;
backward_single_params.inner_itt=100;
backward_single_params.nmax = inf;
backward_single_params.nmin = params2.RI_bg;
backward_single_params.step = 1e-3;
backward_single_params.tv_param = 5e-3;
backward_single_params.use_GPU = use_GPU;
single_solver=BACKWARD_SOLVER_SINGLE(backward_single_params);
[mask, OTF] = single_solver.mask_theoretical_OTF(RI_PEPSI, params2.NA, params2.NA);

display('    Fast TV');

tic;
%profile on;

params=TV.get_default_parameters();
params.outer_itterations=100;
params.inner_itterations=50;
params.min_real = 0;
TV_param = 0.05;
params.TV_strength=TV_param*2;
% params.step = 1e-3;

regulariser=TV(params);

data=single(RI2pot(RI_PEPSI));
data=regulariser.solve(data,mask);

data=pot2RI(data);

%profile viewer;
%profile off;
toc;

clear regulariser;

figure; orthosliceViewer(cat(2, RI_PEPSI,data));
% colormap inferno;caxis(color_disp_range);

% RI_tv=imresize3(real((gather(data_big))),round(size(data_big).*[1 1 resolution(3)/resolution(1)]));


