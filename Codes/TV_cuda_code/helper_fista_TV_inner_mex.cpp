/**
 * @file pctdemo_life_mex_texture.cpp
 * @brief MEX gateway for a stencil operation.
 * Copyright 2013 The MathWorks, Inc.
 *
 */

#include "tmwtypes.h"
#include "mex.h"
#include "gpu/mxGPUArray.h"
#include "helper_fista_TV_inner_mex.hpp"
#include<stdint.h>

#include <cuComplex.h>

 /**
  * MEX gateway
  */
void mexFunction(int /* nlhs */, mxArray *plhs[],
	int nrhs, mxArray const *prhs[])
{

	//INPUTS : input mattrix , lambda , non_neg , inner_itt (check the matlab code for the function of each input)

	char const * const errId = "parallel:gpu:pctdemo_life_mex:InvalidInput";
	char const * const errMsg = "Provide input mattrix , lambda , non_neg , dirichlet_boundary , inner_itt to MEX file.";
	char const * const errMsg3 = "Please use gpuArray for input matrix and single for lambda and bool for non_neg and int32 for inner_itt";
	char const * const errMsg2 = "Check matrix dimenssions. Requirements :  -matt- -> 2 or 3 ;";
	char const * const errMsg4 = "Unknown error -,-";
	char const * const errMsg5 = "Not implemented for complex array.";
	char const * const errMsg6 = "Nonononono";

	// Initialize the MathWorks GPU API.
	mxInitGPU();

	if (nrhs != 10) {
		mexErrMsgIdAndTxt(errId, errMsg);
	}

	// We expect to receive as input an initial board, consisting of CPU data of
	// MATLAB class 'logical', and a scalar double specifying the number of
	// generations to compute.

	for (int i = 0; i < nrhs; ++i) {
		if (i == 0) {
			if (!mxIsGPUArray(prhs[i])) {
				mexErrMsgIdAndTxt(errId, errMsg3);
			}
		}
		else {
			if (mxIsGPUArray(prhs[i])) {
				mexErrMsgIdAndTxt(errId, errMsg3);
			}
		}
	}


	//input mattrix
	mxGPUArray const * const matt = mxGPUCreateFromMxArray(prhs[0]);
	mxComplexity const cmatt = mxGPUGetComplexity(matt);
	mxClassID const tmatt = mxGPUGetClassID(matt);
	mwSize const dmatt = mxGPUGetNumberOfDimensions(matt);
	mwSize const * const smatt = mxGPUGetDimensions(matt);
	float const *  pmatt = nullptr;
	cuFloatComplex const*  pmatt_complex = nullptr;
	if (cmatt == mxCOMPLEX) {
		mexPrintf("TV for complex value --> SLOW");
		pmatt_complex = static_cast<cuFloatComplex const*>(mxGPUGetDataReadOnly(matt));
	}
	else {
		
		pmatt = static_cast<float const*>(mxGPUGetDataReadOnly(matt));
	}
	//lambda
	mxArray const* const param1 = (prhs[1]);
	if (!mxIsSingle(param1)) {
		mexErrMsgIdAndTxt(errId, errMsg3);
	}
	size_t size_lambda=mxGetNumberOfElements(param1);
	if (size_lambda!=1) {
		mexErrMsgIdAndTxt(errId, errMsg3);
	}
	float const* const pparam1 = static_cast<float const*>(mxGetData(param1));
	float lambda = pparam1[0];
	//is_real
	mxArray const* const param2 = (prhs[2]);
	if (!mxIsLogicalScalar(param2)) {
		mexErrMsgIdAndTxt(errId, errMsg3);
	}
	bool is_real = mxIsLogicalScalarTrue(param2);
	//dirichlet_boundary
	mxArray const* const param4 = (prhs[3]);
	if (!mxIsLogicalScalar(param4)) {
		mexErrMsgIdAndTxt(errId, errMsg3);
	}
	bool dirichlet_boundary = mxIsLogicalScalarTrue(param4);
	
	//inner_itt
	mxArray const* const param3 = (prhs[4]);
	if (!mxIsUint32(param3)) {
		mexErrMsgIdAndTxt(errId, errMsg3);
	}
	size_t size_inner_itt = mxGetNumberOfElements(param3);
	if (size_inner_itt != 1) {
		mexErrMsgIdAndTxt(errId, errMsg3);
	}
	uint32_t const* const pparam3 = static_cast<uint32_t const*>(mxGetData(param3));
	uint32_t inner_itt_original = pparam3[0];
	int inner_itt = (int)(inner_itt_original);
	//gather(single(min_real)),gather(single(max_real)),gather(single(min_imag)),gather(single(max_imag)),gather(single(boundary_value))
	// 
	mxArray const* const param42 = (prhs[5]);
	if (!mxIsSingle(param42)) {
		mexErrMsgIdAndTxt(errId, errMsg3);
	}
	size_t size_min_real = mxGetNumberOfElements(param4);
	if (size_min_real != 1) {
		mexErrMsgIdAndTxt(errId, errMsg3);
	}
	float const* const pparam42 = static_cast<float const*>(mxGetData(param42));
	float min_real = pparam42[0];
	//
	mxArray const* const param5 = (prhs[6]);
	if (!mxIsSingle(param5)) {
		mexErrMsgIdAndTxt(errId, errMsg3);
	}
	size_t size_max_real = mxGetNumberOfElements(param5);
	if (size_max_real != 1) {
		mexErrMsgIdAndTxt(errId, errMsg3);
	}
	float const* const pparam5 = static_cast<float const*>(mxGetData(param5));
	float max_real = pparam5[0];
	//
	mxArray const* const param6 = (prhs[7]);
	if (!mxIsSingle(param6)) {
		mexErrMsgIdAndTxt(errId, errMsg3);
	}
	size_t size_min_imag = mxGetNumberOfElements(param6);
	if (size_min_imag != 1) {
		mexErrMsgIdAndTxt(errId, errMsg3);
	}
	float const* const pparam6 = static_cast<float const*>(mxGetData(param6));
	float min_imag = pparam6[0];
	//
	mxArray const* const param7 = (prhs[8]);
	if (!mxIsSingle(param7)) {
		mexErrMsgIdAndTxt(errId, errMsg3);
	}
	size_t size_max_imag = mxGetNumberOfElements(param7);
	if (size_max_imag != 1) {
		mexErrMsgIdAndTxt(errId, errMsg3);
	}
	float const* const pparam7 = static_cast<float const*>(mxGetData(param7));
	float max_imag = pparam7[0];
	//
	mxArray const* const param8 = (prhs[9]);
	if (!mxIsSingle(param8)) {
		mexErrMsgIdAndTxt(errId, errMsg3);
	}
	size_t size_dirichlet_val = mxGetNumberOfElements(param8);
	if (size_dirichlet_val != 1) {
		mexErrMsgIdAndTxt(errId, errMsg3);
	}
	float const*  pparam8 = nullptr ;
	float dirichlet_val;
	cuFloatComplex const* pparam8_complex=nullptr;
	cuFloatComplex dirichlet_val_complex;
	pparam8 = static_cast<float const*>(mxGetData(param8));
	dirichlet_val = pparam8[0];
	if (mxIsComplex(param8)) {
		pparam8_complex = static_cast<cuFloatComplex const*>(mxGetData(param8));
		dirichlet_val_complex = pparam8_complex[0];
	}
	else {
		dirichlet_val_complex = make_cuFloatComplex(dirichlet_val, 0);
	}
	
	//check for the types
	if (tmatt != mxSINGLE_CLASS) {
		mexErrMsgIdAndTxt(errId, errMsg);
	}

	//check for complexity
	if (cmatt == mxCOMPLEX) {
		//mexErrMsgIdAndTxt(errId, errMsg5);
	}
	//check for dimentions
	if (dmatt != 3 && dmatt != 2 ) {
		mexErrMsgIdAndTxt(errId, errMsg2);
	}
	

	// Create two workspace gpuArrays, square real arrays of the same size as
	// the input containing logical data. We will fill these with data, so leave
	// them uninitialized.
	mxGPUArray *  outArray = mxGPUCreateGPUArray(dmatt, smatt,
		mxSINGLE_CLASS, cmatt,
		MX_GPU_INITIALIZE_VALUES);

	float * const pOutArray = static_cast<float *>(mxGPUGetData(outArray));
	cuFloatComplex * pOutArray_complex = nullptr;
	if (cmatt == mxCOMPLEX) {
		pOutArray_complex = static_cast<cuFloatComplex*>(mxGPUGetData(outArray));
	}

	int dims[3] = { (size_t)(smatt[0]) ,(size_t)(smatt[1]),1 };
	if (dmatt == 3) {
		dims[2] = (size_t)(smatt[2]);
	}
	bool complex = false;

	if (cmatt == mxCOMPLEX) { complex = true; }

	if (!complex) {
		int res = MAIN_KERNEL<float>(pmatt, pOutArray, lambda,  dirichlet_boundary, inner_itt, dims, is_real, min_real, max_real, min_imag, max_imag, dirichlet_val);
	}
	else {
		int res = MAIN_KERNEL<cuFloatComplex>(pmatt_complex, pOutArray_complex, lambda, dirichlet_boundary, inner_itt, dims, is_real, min_real, max_real, min_imag, max_imag, dirichlet_val_complex);
	}


	// Wrap the appropriate workspace up as a MATLAB gpuArray for return.

	plhs[0] = mxGPUCreateMxArrayOnGPU(outArray);

	// The mxGPUArray pointers are host-side structures that refer to device
	// data. These must be destroyed before leaving the MEX function.
	mxGPUDestroyGPUArray(matt);
	mxGPUDestroyGPUArray(outArray);
}
