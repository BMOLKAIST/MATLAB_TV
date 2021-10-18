/* Copyright 2013 The MathWorks, Inc. */
//#define DEBUG

#ifndef PCTDEMO_LIFE_SHMEM_HPP
#define PCTDEMO_LIFE_SHMEM_HPP

//template<typename T> int MAIN_KERNEL(T const* const pmatt, T* const pOutArray, float const lambda, bool const non_neg,bool const dirichlet_boundary, int const inner_itt, int const* const dims);
template<typename T> int MAIN_KERNEL(T const* const pmatt, T* const pOutArray, float const lambda, bool const dirichlet_boundary, int const inner_itt, int const* const dims,
	bool const is_real, float const min_real, float const max_real, float const min_imag, float const max_imag, T const dirichlet_val);
#endif
