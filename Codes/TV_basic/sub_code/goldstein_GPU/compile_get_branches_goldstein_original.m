clc;
cd('D:\Projects_\_BORN_CONVERGENT\_multiple_scattering_mshh_v2\sub_code\goldstein_GPU');
mex -output get_branches_goldstein_original get_branches_goldstein.cpp get_branches_goldstein_compute_original.cpp get_branches_goldstein_update.cpp;