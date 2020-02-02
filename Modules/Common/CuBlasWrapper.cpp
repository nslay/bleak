/*-
 * Copyright (c) 2020 Nathan Lay (enslay@gmail.com)
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR(S) ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE AUTHOR(S) BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <stdexcept>
#include <string>
#include <iostream>
#include "CuBlasWrapper.h"

// CUDA stuff
#include "cuda_runtime.h"
#include "cublas_v2.h"

namespace bleak {
namespace gpu_blas {

namespace {

#ifdef _OPENMP
// XXX: Assumed that OpenMP threads operate with a threadpool
bool g_bInitialized = false;
cublasHandle_t g_handle = cublasHandle_t();
#pragma omp threadprivate(g_handle, g_bInitialized)
#else // !_OPENMP
thread_local bool g_bInitialized = false;
thread_local cublasHandle_t g_handle = cublasHandle_t();
#endif // _OPENMP

cublasOperation_t GetTrans(char trans) {
  switch (trans) {
  case 'n':
  case 'N':
    return CUBLAS_OP_N;
  case 'c':
  case 'C':
  case 't':
  case 'T':
    return CUBLAS_OP_T;
  }

  std::cerr << "Error: Invalid op '" << trans << "'." << std::endl;
  throw std::runtime_error(std::string("Error: Invalid op '") + trans + "'.");
  // Not reached
}

} // end anonymous namespace

// Setup per-thread context
void Initialize() {
  // XXX: This should be called after threads are bound to their specific devices!

  if (g_bInitialized)
    return;

  if (cublasCreate(&g_handle) != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "Error: Failed to create cuBLAS handle." << std::endl;
    throw std::runtime_error("Error: Failed to create cuBLAS handle.");
  }

  g_bInitialized = true;
}

#define CheckCuBLASCall(func, ...) if (func(__VA_ARGS__) != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("Error: cuBLAS " #func " failed.")

// Level 1

template<>
void swap<float>(int n, float *x, int incx, float *y, int incy) { CheckCuBLASCall(cublasSswap, g_handle, n, x, incx, y, incy); }

template<>
void swap<double>(int n, double *x, int incx, double *y, int incy) { CheckCuBLASCall(cublasDswap, g_handle, n, x, incx, y, incy); }

template<>
void scal<float>(int n, const float &a, float *x, int incx) { CheckCuBLASCall(cublasSscal, g_handle, n, &a, x, incx); }

template<>
void scal<double>(int n, const double &a, double *x, int incx) { CheckCuBLASCall(cublasDscal, g_handle, n, &a, x, incx); }

template<>
void copy<float>(int n, const float *x, int incx, float *y, int incy) { CheckCuBLASCall(cublasScopy, g_handle, n, x, incx, y, incy); }

template<>
void copy<double>(int n, const double *x, int incx, double *y, int incy) { CheckCuBLASCall(cublasDcopy, g_handle, n, x, incx, y, incy); }

template<>
void axpy<float>(int n, const float &a, const float *x, int incx, float *y, int incy) { CheckCuBLASCall(cublasSaxpy, g_handle, n, &a, x, incx, y, incy); }

template<>
void axpy<double>(int n, const double &a, const double *x, int incx, double *y, int incy) { CheckCuBLASCall(cublasDaxpy, g_handle, n, &a, x, incx, y, incy); }

template<>
float dot<float>(int n, const float *x, int incx, const float *y, int incy) { 
  float result = 0.0f;
  CheckCuBLASCall(cublasSdot, g_handle, n, x, incx, y, incy, &result);
  return result;
}

template<>
double dot<double>(int n, const double *x, int incx, const double *y, int incy) { 
  double result = 0.0;
  CheckCuBLASCall(cublasDdot, g_handle, n, x, incx, y, incy, &result);
  return result;
}

template<>
void dot<float>(int n, const float *x, int incx, const float *y, int incy, float &result) { CheckCuBLASCall(cublasSdot, g_handle, n, x, incx, y, incy, &result); }

template<>
void dot<double>(int n, const double *x, int incx, const double *y, int incy, double &result) { CheckCuBLASCall(cublasDdot, g_handle, n, x, incx, y, incy, &result); }

template<>
float nrm2<float>(int n, const float *x, int incx) {
  float result = 0.0f;
  CheckCuBLASCall(cublasSnrm2, g_handle, n, x, incx, &result);
  return result;
}

template<>
double nrm2<double>(int n, const double *x, int incx) {
  double result = 0.0;
  CheckCuBLASCall(cublasDnrm2, g_handle, n, x, incx, &result);
  return result;
}

template<>
void nrm2<float>(int n, const float *x, int incx, float &result) { CheckCuBLASCall(cublasSnrm2, g_handle, n, x, incx, &result); }

template<>
void nrm2<double>(int n, const double *x, int incx, double &result) { CheckCuBLASCall(cublasDnrm2, g_handle, n, x, incx, &result); }

template<>
float asum<float>(int n, const float *x, int incx) {
  float result = 0.0f;
  CheckCuBLASCall(cublasSasum, g_handle, n, x, incx, &result);
  return result;
}

template<>
double asum<double>(int n, const double *x, int incx) {
  double result = 0.0;
  CheckCuBLASCall(cublasDasum, g_handle, n, x, incx, &result);
  return result;
}

template<>
void asum<float>(int n, const float *x, int incx, float &result) { CheckCuBLASCall(cublasSasum, g_handle, n, x, incx, &result); }

template<>
void asum<double>(int n, const double *x, int incx, double &result) { CheckCuBLASCall(cublasDasum, g_handle, n, x, incx, &result); }

template<>
int amax<float>(int n, const float *x, int incx) {
  int result = -1;
  CheckCuBLASCall(cublasIsamax, g_handle, n, x, incx, &result);
  return result;
}

template<>
int amax<double>(int n, const double *x, int incx) {
  int result = -1;
  CheckCuBLASCall(cublasIdamax, g_handle, n, x, incx, &result);
  return result;
}

template<>
void amax<float>(int n, const float *x, int incx, int &result) { CheckCuBLASCall(cublasIsamax, g_handle, n, x, incx, &result); }

template<>
void amax<double>(int n, const double *x, int incx, int &result) { CheckCuBLASCall(cublasIdamax, g_handle, n, x, incx, &result); }

// Level 2
template<>
void gemv<float>(char trans, int m, int n, const float &alpha, const float *a, int lda, const float *x, int incx, const float &beta, float *y, int incy) {
  CheckCuBLASCall(cublasSgemv, g_handle, GetTrans(trans), m, n, &alpha, a, lda, x, incx, &beta, y, incy);
}

template<>
void gemv<double>(char trans, int m, int n, const double &alpha, const double *a, int lda, const double *x, int incx, const double &beta, double *y, int incy) {
  CheckCuBLASCall(cublasDgemv, g_handle, GetTrans(trans), m, n, &alpha, a, lda, x, incx, &beta, y, incy);
}

template<> 
void ger<float>(int m, int n, const float &alpha, const float *x, int incx, const float *y, int incy, float *a, int lda) {
  CheckCuBLASCall(cublasSger, g_handle, m, n, &alpha, x, incx, y, incy, a, lda);
}

template<> 
void ger<double>(int m, int n, const double &alpha, const double *x, int incx, const double *y, int incy, double *a, int lda) {
  CheckCuBLASCall(cublasDger, g_handle, m, n, &alpha, x, incx, y, incy, a, lda);
}

// Level 3
template<>
void gemm<float>(char transa, char transb, int m, int n, int k, const float &alpha, const float *a, int lda, const float *b, int ldb, const float &beta, float *c, int ldc) {
  CheckCuBLASCall(cublasSgemm, g_handle, GetTrans(transa), GetTrans(transb), m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc);
}

template<>
void gemm<double>(char transa, char transb, int m, int n, int k, const double &alpha, const double *a, int lda, const double *b, int ldb, const double &beta, double *c, int ldc) {
  CheckCuBLASCall(cublasDgemm, g_handle, GetTrans(transa), GetTrans(transb), m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc);
}

} // end namespace gpu_blas
} // end namespace bleak 
