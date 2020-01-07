/*-
 * Copyright (c) 2019 Nathan Lay (enslay@gmail.com)
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

#ifndef BLASWRAPPER_H
#define BLASWRAPPER_H

// This is a ZERO-INDEXED COLUMN-MAJOR wrapper to BLAS
// All type prefixes removed from function names since it's templated...

namespace bleak {
namespace cpu_blas {

// Setup per-thread context
void Initialize();

// Level 1
template<typename RealType>
void swap(int n, RealType *x, int incx, RealType *y, int incy);

template<typename RealType>
void scal(int n, const RealType &a, RealType *x, int incx);

template<typename RealType>
void copy(int n, const RealType *x, int incx, RealType *y, int incy);

template<typename RealType>
void axpy(int n, const RealType &a, const RealType *x, int incx, RealType *y, int incy);

template<typename RealType>
RealType dot(int n, const RealType *x, int incx, const RealType *y, int incy);

template<typename RealType>
RealType nrm2(int n, const RealType *x, int incx);

template<typename RealType>
RealType asum(int n, const RealType *x, int incx);

template<typename RealType>
int amax(int n, const RealType *x, int incx);

// Level 2
template<typename RealType>
void gemv(char trans, int m, int n, const RealType &alpha, const RealType *a, int lda, const RealType *x, int incx, const RealType &beta, RealType *y, int incy);

// Level 3
template<typename RealType>
void gemm(char transa, char transb, int m, int n, int k, const RealType &alpha, const RealType *a, int lda, const RealType *b, int ldb, const RealType &beta, RealType *c, int ldc);

} // end namespace cpu_blas
} // end namespace bleak

// This is the default bleak BLAS (it sucks!). Will later wrap other hard-to-compile-on-Windows BLAS in the future!
// But it's inlined templated, so hopefully some compiler optimization will come of it?
#include "SlowBlas.h"

#endif // !BLASWRAPPER_H
