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

#pragma once

#ifndef BLEAK_OPENBLASWRAPPER_H
#define BLEAK_OPENBLASWRAPPER_H

#include "BlasWrapper.h"
#include "cblas.h"

namespace bleak {
namespace cpu_blas {

inline void Initialize() { } // Nothing to do...

// Level 1
template<>
inline void swap<float>(int n, float *x, int incx, float *y, int incy) { cblas_sswap(n, x, incx, y, incy); }

template<>
inline void swap<double>(int n, double *x, int incx, double *y, int incy) { cblas_dswap(n, x, incx, y, incy); }

template<>
inline void scal<float>(int n, const float &a, float *x, int incx) { cblas_sscal(n, a, x, incx); }

template<>
inline void scal<double>(int n, const double &a, double *x, int incx) { cblas_dscal(n, a, x, incx); }

template<>
inline void copy<float>(int n, const float *x, int incx, float *y, int incy) { return cblas_scopy(n, x, incx, y, incy); }

template<>
inline void copy<double>(int n, const double *x, int incx, double *y, int incy) { return cblas_dcopy(n, x, incx, y, incy); }

template<>
inline void axpy<float>(int n, const float &a, const float *x, int incx, float *y, int incy) { return cblas_saxpy(n, a, x, incx, y, incy); }

template<>
inline void axpy<double>(int n, const double &a, const double *x, int incx, double *y, int incy) { return cblas_daxpy(n, a, x, incx, y, incy); }

template<>
inline float dot<float>(int n, const float *x, int incx, const float *y, int incy) { return cblas_sdot(n, x, incx, y, incy); }

template<>
inline double dot<double>(int n, const double *x, int incx, const double *y, int incy) { return cblas_ddot(n, x, incx, y, incy); }

template<>
inline float nrm2<float>(int n, const float *x, int incx) { return cblas_snrm2(n, x, incx); }

template<>
inline double nrm2<double>(int n, const double *x, int incx) { return cblas_dnrm2(n, x, incx); }

template<>
inline float asum<float>(int n, const float *x, int incx) { return cblas_sasum(n, x, incx); }

template<>
inline double asum<double>(int n, const double *x, int incx) { return cblas_dasum(n, x, incx); }

template<>
inline int amax<float>(int n, const float *x, int incx) { return (int)cblas_isamax(n, x, incx); }

template<>
inline int amax<double>(int n, const double *x, int incx) { return (int)cblas_idamax(n, x, incx); }

// Level 2
extern template void gemv<float>(char trans, int m, int n, const float &alpha, const float *a, int lda, const float *x, int incx, const float &beta, float *y, int incy);
extern template void gemv<double>(char trans, int m, int n, const double &alpha, const double *a, int lda, const double *x, int incx, const double &beta, double *y, int incy);

// Level 3
extern template void gemm<float>(char transa, char transb, int m, int n, int k, const float &alpha, const float *a, int lda, const float *b, int ldb, const float &beta, float *c, int ldc);
extern template void gemm<double>(char transa, char transb, int m, int n, int k, const double &alpha, const double *a, int lda, const double *b, int ldb, const double &beta, double *c, int ldc);

} // end namespace cpu_blas
} // end namespace bleak

#endif // BLEAK_OPENBLASWRAPPER_H
