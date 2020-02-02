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

#include <cstring>
#include <cstdlib>
#include <stdexcept>
#include <iostream>
#include "OpenBlasWrapper.h"

namespace bleak {
namespace cpu_blas {

namespace {

CBLAS_TRANSPOSE GetTrans(char trans) {
  switch (trans) {
  case 'n':
  case 'N':
    return CblasNoTrans;
  case 'c':
  case 'C':
  case 't':
  case 'T':
    return CblasTrans;
  }

  std::cerr << "Error: Invalid op '" << trans << "'." << std::endl;
  throw std::runtime_error(std::string("Error: Invalid op '") + trans + "'.");
  // Not reached
}

} // end anonymous namespace

void Initialize() { 
  switch (openblas_get_parallel()) {
  case OPENBLAS_SEQUENTIAL:
    break;
  case OPENBLAS_THREAD:
    {
      int iNumThreads = openblas_get_num_procs();
      const char * const p_cNumThreads = getenv("OMP_NUM_THREADS"); // Let's respect this variable anyway...
      if (p_cNumThreads != nullptr && p_cNumThreads[0] != '\0') {
        char *p = nullptr;
        const int iTmp = (int)strtoul(p_cNumThreads, &p, 10);
        if (*p == '\0' && iTmp > 0)
          iNumThreads = iTmp;
      }
      openblas_set_num_threads(iNumThreads);
    }
    break;
  case OPENBLAS_OPENMP:
    // OMP_NUM_THREADS is already respected
    break;
  default:
    break;
  }
}

// Level 2
template<>
void gemv<float>(char trans, int m, int n, const float &alpha, const float *a, int lda, const float *x, int incx, const float &beta, float *y, int incy) { 
  cblas_sgemv(CblasColMajor, GetTrans(trans), m, n, alpha, a, lda, x, incx, beta, y, incy); 
}

template<>
void gemv<double>(char trans, int m, int n, const double &alpha, const double *a, int lda, const double *x, int incx, const double &beta, double *y, int incy) { 
  cblas_dgemv(CblasColMajor, GetTrans(trans), m, n, alpha, a, lda, x, incx, beta, y, incy); 
}

template<>
void ger<float>(int m, int n, const float &alpha, const float *x, int incx, const float *y, int incy, float *a, int lda) {
  cblas_sger(CblasColMajor, m, n, alpha, x, incx, y, incy, a, lda);
}

template<>
void ger<double>(int m, int n, const double &alpha, const double *x, int incx, const double *y, int incy, double *a, int lda) {
  cblas_dger(CblasColMajor, m, n, alpha, x, incx, y, incy, a, lda);
}

// Level 3
template<>
void gemm<float>(char transa, char transb, int m, int n, int k, const float &alpha, const float *a, int lda, const float *b, int ldb, const float &beta, float *c, int ldc) {
  cblas_sgemm(CblasColMajor, GetTrans(transa), GetTrans(transb), m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

template<>
void gemm<double>(char transa, char transb, int m, int n, int k, const double &alpha, const double *a, int lda, const double *b, int ldb, const double &beta, double *c, int ldc) {
  cblas_dgemm(CblasColMajor, GetTrans(transa), GetTrans(transb), m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

} // end namespace cpu_blas
} // end namespace bleak
