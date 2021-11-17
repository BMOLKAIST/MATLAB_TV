#ifndef COMPLEX_OP
#define COMPLEX_OP

/**
 * Original code by Travis W. Thompson
 */
#include <cuComplex.h>

__host__ __device__
cuFloatComplex operator+(cuFloatComplex a, cuFloatComplex b) { return cuCaddf(a, b); }

template <class T>
__host__ __device__
cuFloatComplex operator+(cuFloatComplex a, T b) { return cuCaddf(a, make_cuFloatComplex(b, 0)); }

template <class T>
__host__ __device__
cuFloatComplex operator+(T a, cuFloatComplex b) { return cuCaddf(make_cuFloatComplex(a, 0), b); }

//-----------------

__host__ __device__
cuFloatComplex operator-(cuFloatComplex a, cuFloatComplex b) { return cuCsubf(a, b); }

template <class T>
__host__ __device__
cuFloatComplex operator-(cuFloatComplex a, T b) { return cuCsubf(a, make_cuFloatComplex(b, 0)); }

template <class T>
__host__ __device__
cuFloatComplex operator-(T a, cuFloatComplex b) { return cuCsubf(make_cuFloatComplex(a, 0), b); }

//-----------------

__host__ __device__
cuFloatComplex operator*(cuFloatComplex a, cuFloatComplex b) { return cuCmulf(a, b); }

template <class T>
__host__ __device__
cuFloatComplex operator*(cuFloatComplex a, T b) { return cuCmulf(a, make_cuFloatComplex(b, 0)); }

template <class T>
__host__ __device__
cuFloatComplex operator*(T a, cuFloatComplex b) { return cuCmulf(make_cuFloatComplex(a, 0), b); }

__host__ __device__
cuFloatComplex operator/(cuFloatComplex a, cuFloatComplex b) { return cuCdivf(a, b); }

template <class T>
__host__ __device__
cuFloatComplex operator/(cuFloatComplex a, T b) { return cuCdivf(a, make_cuFloatComplex(b, 0)); }

template <class T>
__host__ __device__
cuFloatComplex operator/(T a, cuFloatComplex b) { return cuCdivf(make_cuFloatComplex(a, 0), b); }

//-----------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------

__host__ __device__
cuDoubleComplex operator+(cuDoubleComplex a, cuDoubleComplex b) { return cuCadd(a, b); }

template <class T>
__host__ __device__
cuDoubleComplex operator+(cuDoubleComplex a, T b) { return cuCadd(a, make_cuDoubleComplex(b, 0)); }

template <class T>
__host__ __device__
cuDoubleComplex operator+(T a, cuDoubleComplex b) { return cuCadd(make_cuDoubleComplex(a, 0), b); }

//-----------------

__host__ __device__
cuDoubleComplex operator-(cuDoubleComplex a, cuDoubleComplex b) { return cuCsub(a, b); }

template <class T>
__host__ __device__
cuDoubleComplex operator-(cuDoubleComplex a, T b) { return cuCsub(a, make_cuDoubleComplex(b, 0)); }

template <class T>
__host__ __device__
cuDoubleComplex operator-(T a, cuDoubleComplex b) { return cuCsub(make_cuDoubleComplex(a, 0), b); }

//-----------------

__host__ __device__
cuDoubleComplex operator*(cuDoubleComplex a, cuDoubleComplex b) { return cuCmul(a, b); }

template <class T>
__host__ __device__
cuDoubleComplex operator*(cuDoubleComplex a, T b) { return cuCmul(a, make_cuDoubleComplex(b, 0)); }

template <class T>
__host__ __device__
cuDoubleComplex operator*(T a, cuDoubleComplex b) { return cuCmul(make_cuDoubleComplex(a, 0), b); }

//-----------------

__host__ __device__
cuDoubleComplex operator/(cuDoubleComplex a, cuDoubleComplex b) { return cuCdiv(a, b); }

template <class T>
__host__ __device__
cuDoubleComplex operator/(cuDoubleComplex a, T b) { return cuCdiv(a, make_cuDoubleComplex(b, 0)); }

template <class T>
__host__ __device__
cuDoubleComplex operator/(T a, cuDoubleComplex b) { return cuCdiv(make_cuDoubleComplex(a, 0), b); }

#endif













