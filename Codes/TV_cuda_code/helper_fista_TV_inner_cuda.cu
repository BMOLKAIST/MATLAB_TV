
#include <algorithm>
#include <math.h>
#include <cuda_runtime_api.h>
#include "helper_fista_TV_inner_mex.hpp"
#include "mex.h"
#include <cuComplex.h>
#include "cuComplex_Op.h"
#include <type_traits>


__device__
int smaller_to_end(int value, int end) {
	return (value<0) ? end : value;
}
__device__
int higher_to_begin(int value, int end) {
	return (value>end) ? 0 : value;
}

__device__
void complex_mult(const float & a_1, const float & a_2, const float & b_1, const float & b_2, float* c_1, float* c_2) {
	*c_1 = a_1 * b_1 - a_2 * b_2;
	*c_2 = a_1 * b_2 + a_2 * b_1;
}
__device__
float norm3(const float& a, const float& b, const float& c) {
	return sqrt((a * a + b * b + c * c));
}
__device__
float norm3(const cuFloatComplex& a, const cuFloatComplex& b, const cuFloatComplex& c) {
	return norm3(cuCabsf(a), cuCabsf(b), cuCabsf(c));
}

template<typename T>
__global__
void kernel_1(T const * const in_matt, T const * const R_matt, T * const A_matt, T* const A_tmp_matt,
	int const dim_1, int const dim_2, int const dim_3,float const lambda,bool const dirichlet_boundary,
	bool const is_real, float const min_real, float const max_real, float const min_imag, float const max_imag, T const dirichlet_val)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	int d3 = id / (dim_1 * dim_2);
	//int d12 = id % (dim_1* dim_2);
	int d12 = id - (d3*dim_1 * dim_2);
	int d2 = d12 / dim_1;
	//int d1 = d12 % dim_1;
	int d1 = d12 - (dim_1 * d2);
	
	//carefull because of complex number dimentions are twice big
	
	//verify if in the 3D matrix 
	if (d1 < dim_1 && d2 < dim_2 && d3 < dim_3) {

		T A_val = A_tmp_matt[id];

		A_val = A_val - R_matt[smaller_to_end(d1 - 1, dim_1 - 1) + (d2)*dim_1 + (d3)*dim_1 * dim_2];
		A_val = A_val - R_matt[d1 + smaller_to_end(d2 - 1, dim_2 - 1) * dim_1 + (d3)*dim_1 * dim_2 + dim_1 * dim_2 * dim_3];
		A_val = A_val - R_matt[d1 + (d2)*dim_1 + smaller_to_end(d3 - 1, dim_3 - 1) * dim_1 * dim_2 + dim_1 * dim_2 * dim_3 * 2];

		A_val = in_matt[id] - lambda * A_val;

		non_neg(A_val, min_real, max_real, min_imag, max_imag);
		//A_val = (A_val > min_real ) ? A_val : min_real;
		//A_val = (A_val < max_real ) ? A_val : max_real;
		

		A_val = (dirichlet_boundary && (d1==0 || d2 == 0 || d3 == 0 || d1 == dim_1-1 || d2 == dim_2-1 || d3 == dim_3-1)) ? dirichlet_val : A_val ;

		A_matt[id] = A_val;
	}
	
}
template<typename T>
__device__
void non_neg(T& A_val, float const min_real, float const max_real, float const min_imag, float const max_imag) {
	A_val = (A_val > min_real) ? A_val : min_real;
	A_val = (A_val < max_real) ? A_val : max_real;
}
template<>
__device__
void non_neg<cuFloatComplex>(cuFloatComplex& A_val, float const min_real, float const max_real, float const min_imag, float const max_imag) {
	A_val = (cuCrealf(A_val) > min_real) ? A_val : make_cuFloatComplex(min_real,cuCimagf(A_val));
	A_val = (cuCrealf(A_val) < max_real) ? A_val : make_cuFloatComplex(max_real,cuCimagf(A_val));

	A_val = (cuCimagf(A_val) > min_real) ? A_val : make_cuFloatComplex(cuCrealf(A_val), min_imag);
	A_val = (cuCimagf(A_val) < max_real) ? A_val : make_cuFloatComplex(cuCrealf(A_val), max_imag);
}

template<typename T>
__device__
void set_zero(T& val) {
	val = 0;
}
template<>
__device__
void set_zero<cuFloatComplex>(cuFloatComplex& val) {
	val = make_cuFloatComplex(0,0);
}


template<typename T>
__global__
void kernel_2(T const* const in_matt, T * const R_matt, T* const Pold_matt, T * const A_matt, T* const A_tmp_matt,
	int const dim_1, int const dim_2, int const dim_3, float const lambda, float const step,int const action_flag)
{
	
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	int d3 = id / (dim_1 * dim_2);
	//int d12 = id % (dim_1* dim_2);
	int d12 = id - (d3 * dim_1 * dim_2);
	int d2 = d12 / dim_1;
	//int d1 = d12 % dim_1;
	int d1 = d12 - (dim_1 * d2);

	//carefull because of complex number dimentions are twice big

	//verify if in the 3D matrix 
	if (d1 < dim_1 && d2 < dim_2 && d3 < dim_3) {
		float dividend = (dim_3 == 1) ? 8 : 12;

		dividend = 1 / (dividend * lambda);

		T R1; set_zero(R1);
		T R2; set_zero(R2);
		T R3; set_zero(R3);


		//TV_L_trans
		T A_0 = A_matt[id];
		R1 = A_0 - A_matt[higher_to_begin(d1 + 1, dim_1 - 1) + (d2)*dim_1 + (d3)*dim_1 * dim_2];
		R2 = A_0 - A_matt[(d1)+higher_to_begin(d2 + 1, dim_2 - 1) * dim_1 + (d3)*dim_1 * dim_2];
		R3 = A_0 - A_matt[(d1)+(d2)*dim_1 + higher_to_begin(d3 + 1, dim_3 - 1) * dim_1 * dim_2];
		//rest
		R1 = dividend * R1 + R_matt[(d1)+(d2)*dim_1 + (d3)*dim_1 * dim_2];
		R2 = dividend * R2 + R_matt[(d1)+(d2)*dim_1 + (d3)*dim_1 * dim_2 + dim_1 * dim_2 * dim_3];
		R3 = dividend * R3 + R_matt[(d1)+(d2)*dim_1 + (d3)*dim_1 * dim_2 + dim_1 * dim_2 * dim_3 * 2];
		//project
		float norm = norm3(R1, R2, R3);
		norm = (norm > 1) ? (1/norm) : 1;

		R1 = R1 * norm;
		R2 = R2 * norm;
		R3 = R3 * norm;

		T temp_var;
		T A_temp;

		float used_step = step;

		if (action_flag == 1) {
			used_step = 0;
		}

		temp_var = R1 + used_step * (R1 - Pold_matt[(d1)+(d2)*dim_1 + (d3)*dim_1 * dim_2]);
		A_temp = temp_var;
		R_matt[(d1)+(d2)*dim_1 + (d3)*dim_1 * dim_2] = temp_var;

		temp_var = R2 + used_step * (R2 - Pold_matt[(d1)+(d2)*dim_1 + (d3)*dim_1 * dim_2 + dim_1 * dim_2 * dim_3]);
		A_temp = A_temp + temp_var;
		R_matt[(d1)+(d2)*dim_1 + (d3)*dim_1 * dim_2 + dim_1 * dim_2 * dim_3] = temp_var;

		temp_var = R3 + used_step * (R3 - Pold_matt[(d1)+(d2)*dim_1 + (d3)*dim_1 * dim_2 + dim_1 * dim_2 * dim_3 * 2]);
		A_temp = A_temp + temp_var;
		R_matt[(d1)+(d2)*dim_1 + (d3)*dim_1 * dim_2 + dim_1 * dim_2 * dim_3 * 2] = temp_var;

		A_tmp_matt[id] = A_temp;

		Pold_matt[(d1)+(d2)*dim_1 + (d3)*dim_1 * dim_2] = R1;
		Pold_matt[(d1)+(d2)*dim_1 + (d3)*dim_1 * dim_2 + dim_1 * dim_2 * dim_3] = R2;
		Pold_matt[(d1)+(d2)*dim_1 + (d3)*dim_1 * dim_2 + dim_1 * dim_2 * dim_3 * 2] = R3;

	}

}

template<typename T> int MAIN_KERNEL(T const * const pmatt, T * const pOutArray, float const lambda,bool const dirichlet_boundary,int const inner_itt, int const * const dims,
	bool const is_real, float const min_real, float const max_real, float const min_imag, float const max_imag, T const dirichlet_val)
{
	//mexPrintf("%f\n", min_real);
	//mexPrintf("%f\n", max_real);
	//put the dimension in simpler variables
	int const dim_1 = dims[0];
	int const dim_2 = dims[1];
	int const dim_3 = dims[2];

	//creat temporary variable 
	
	T* R_matt = nullptr;
	cudaMalloc((void**)&R_matt, 3 * dim_1 * dim_2 * dim_3 * sizeof(T));
	cudaMemset(R_matt, 0, 3 * dim_1 * dim_2 * dim_3 * sizeof(T));
	T* Pold_matt = nullptr;
	cudaMalloc((void**)&Pold_matt, 3 * dim_1 * dim_2 * dim_3 * sizeof(T));
	cudaMemset(Pold_matt, 0, 3 * dim_1 * dim_2 * dim_3 * sizeof(T));
	T* A_tmp_matt = nullptr;
	cudaMalloc((void**)&A_tmp_matt, dim_1 * dim_2 * dim_3 * sizeof(T));
	cudaMemset(A_tmp_matt, 0, dim_1 * dim_2 * dim_3 * sizeof(T));
	

	//variables for launch configuration
	int blockSize_1;
	int minGridSize_1;
	int gridSize_1;
	int blockSize_2;
	int minGridSize_2;
	int gridSize_2;

	int arrayCount_1 = dim_1 * dim_2*dim_3;
	int arrayCount_2 = dim_1 * dim_2 * dim_3;

	cudaOccupancyMaxPotentialBlockSize(&minGridSize_1, &blockSize_1, (void*)kernel_1<T>, 0, arrayCount_1);
	cudaOccupancyMaxPotentialBlockSize(&minGridSize_2, &blockSize_2, (void*)kernel_2<T>, 0, arrayCount_2);
	// Round up according to array size
	gridSize_1 = (arrayCount_1 + blockSize_1 - 1) / blockSize_1;
	gridSize_2 = (arrayCount_2 + blockSize_2 - 1) / blockSize_2;
	
	float t_n = 1;
	float t_np = 1;
	float step = 0;
	int action_flag = -1;
	
	//init A
	kernel_1<T> << <gridSize_1, blockSize_1 >> > (pmatt, R_matt, pOutArray, A_tmp_matt,
		dim_1, dim_2, dim_3, lambda, dirichlet_boundary, is_real, min_real, max_real, min_imag, max_imag, dirichlet_val);

	for (int itt = 0; itt < inner_itt; itt++) {
		t_n = t_np;

		t_np = (1 + sqrt(1 + 4*t_n*t_n)) / 2;
		step = ((t_n - 1) / t_np);

		if (itt == inner_itt - 1) {
			action_flag = 1;
		}

		kernel_2<T> << <gridSize_2, blockSize_2 >> > (pmatt, R_matt, Pold_matt, pOutArray, A_tmp_matt,
			dim_1, dim_2, dim_3, lambda, step, action_flag);

		kernel_1<T> << <gridSize_1, blockSize_1 >> > (pmatt, R_matt, pOutArray, A_tmp_matt,
			dim_1, dim_2, dim_3, lambda, dirichlet_boundary, is_real, min_real, max_real, min_imag, max_imag, dirichlet_val);

		action_flag = 0;

	}
	//cudaDeviceSynchronize();



	// free and return
	cudaFree(R_matt);
	cudaFree(Pold_matt);
	cudaFree(A_tmp_matt);
	return 1;
}

template int MAIN_KERNEL<float>(float const* const pmatt, float* const pOutArray, float const lambda, bool const dirichlet_boundary, int const inner_itt, int const* const dims,
	bool const is_real, float const min_real, float const max_real, float const min_imag, float const max_imag, float const dirichlet_val);
template int MAIN_KERNEL<cuFloatComplex>(cuFloatComplex const* const pmatt, cuFloatComplex* const pOutArray, float const lambda, bool const dirichlet_boundary, int const inner_itt, int const* const dims,
	bool const is_real, float const min_real, float const max_real, float const min_imag, float const max_imag, cuFloatComplex const dirichlet_val);
