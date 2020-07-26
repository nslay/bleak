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

#ifndef BLEAK_SLOWBLAS_H
#define BLEAK_SLOWBLAS_H

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <algorithm>

namespace bleak {
namespace cpu_blas {

// NOTE: This implementation is a stand-in where BLAS is difficult to setup (Windows)
//       At least you can *run* the thing...

// Code loosely translated from reference implementation: http://www.netlib.org/blas/

// Does nothing for this implementation
inline void Initialize() { }

// Level 1
template<typename RealType>
void swap(int n, RealType *x, int incx, RealType *y, int incy) {
  if (n <= 0 || x == nullptr || y == nullptr)
    return;

  if (incx < 0)
    x += -(n-1)*incx;

  if (incy < 0)
    y += -(n-1)*incy;

  if (incx == 1 && incy == 1) {
    std::swap_ranges(x, x + n, y);
  }
  else {
    for (int i = 0; i < n; ++i, x += incx, y += incy)
      std::swap(*x, *y);
  }
}

template<typename RealType>
void scal(int n, const RealType &a, RealType *x, int incx) {
  if (n <= 0 || incx <= 0 || x == nullptr)
    return;

  for (int i = 0; i < n; ++i, x += incx)
    *x *= a;
}

template<typename RealType>
void copy(int n, const RealType *x, int incx, RealType *y, int incy) {
  if (n <= 0 || x == nullptr || y == nullptr)
    return;

  if (incx < 0)
    x += -(n-1)*incx;

  if (incy < 0)
    y += -(n-1)*incy;

  if (incx == 1 && incy == 1) {
    std::copy(x, x + n, y);
  }
  else {
    for (int i = 0; i < n; ++i, x += incx, y += incy)
      *y = *x;
  }
}

template<typename RealType>
void axpy(int n, const RealType &a, const RealType *x, int incx, RealType *y, int incy) {
  if (n <= 0 || x == nullptr || y == nullptr)
    return;

  if (incx < 0)
    x += -(n-1)*incx;

  if (incy < 0)
    y += -(n-1)*incy;

  for (int i = 0; i < n; ++i, x += incx, y += incy)
    *y += a*(*x);
}

template<typename RealType>
RealType dot(int n, const RealType *x, int incx, const RealType *y, int incy) {
  if (n <= 0 || x == nullptr || y == nullptr)
    return RealType(0);

  if (incx < 0)
    x += -(n-1)*incx;

  if (incy < 0)
    y += -(n-1)*incy;

  RealType sum = RealType(0);
  for (int i = 0; i < n; ++i, x += incx, y += incy)
    sum += (*x) * (*y);

  return sum;
}

template<typename RealType>
void dot(int n, const RealType *x, int incx, const RealType *y, int incy, RealType *result) { *result = dot<RealType>(n, x, incx, y, incy); }

template<typename RealType>
RealType nrm2(int n, const RealType *x, int incx) {
  if (n <= 0 || incx <= 0 || x == nullptr)
    return RealType(0);

  if (n == 1)
    return std::abs(*x);

  RealType scale = RealType(0);
  RealType ssq = RealType(1);

  for (int i = 0; i < n; ++i, x += incx) {
    if (*x != RealType(0)) {
      const RealType tmp = std::abs(*x);
      if (scale < *x) {
        ssq = RealType(1) + ssq * RealType(std::pow(scale/tmp, 2));
        scale = tmp;
      }
      else {
        ssq += RealType(std::pow(tmp/scale, 2));
      }
    }
  }

  return scale*std::sqrt(ssq);
}

template<typename RealType>
void nrm2(int n, const RealType *x, int incx, RealType *result) { *result = nrm2<RealType>(n, x, incx); }

template<typename RealType>
RealType asum(int n, const RealType *x, int incx) {
  if (n <= 0 || incx <= 0 || x == nullptr)
    return RealType(0);

  RealType sum = RealType(0);

  for (int i = 0; i < n; ++i, x += incx)
    sum += std::abs(*x);

  return sum;
}

template<typename RealType>
void asum(int n, const RealType *x, int incx, RealType *result) { *result = asum<RealType>(n, x, incx); }

template<typename RealType>
int amax(int n, const RealType *x, int incx) {
  if (n <= 0 || incx <= 0 || x == nullptr)
    return -1;

  if (n == 1)
    return 0;

  RealType max = std::abs(*x);
  int imax = 0;

  x += incx;

  for (int i = 1; i < n; ++i, x += incx) {
    const RealType tmp = std::abs(*x);
    if (tmp > max) {
      max = tmp;
      imax = i;
    }
  }

  return imax;
}

template<typename RealType>
void amax(int n, const RealType *x, int incx, int *result) { *result = amax<RealType>(n, x, incx); }

// Level 2
template<typename RealType>
void gemv(char trans, int m, int n, const RealType &alpha, const RealType *a, int lda, const RealType *x, int incx, const RealType &beta, RealType *y, int incy) {
  if (m < 0 || n < 0 || lda < std::max(1,m) || incx == 0 || incy == 0 || a == nullptr || x == nullptr || y == nullptr)
    return;

  int lenx = 0, leny = 0;
  switch (trans) {
  case 'n':
  case 'N':
    lenx = n;
    leny = m;
    break;
  case 'c':
  case 't':
  case 'C':
  case 'T':
    lenx = m;
    leny = n;
    break;
  default:
    std::cerr << "Error: Invalid op '" << trans << "'." << std::endl;
    throw std::runtime_error(std::string("Error: Invalid op '") + trans + "'.");
  }

  if (incx < 0)
    x += -(lenx-1)*incx;

  if (incy < 0)
    y += -(leny-1)*incy;

  if (beta != RealType(1)) {
    RealType *yt = y;
    if (beta == RealType(0)) {
      for (int i = 0; i < leny; ++i, yt += incy)
        *yt = RealType(0);
    }
    else {
      for (int i = 0; i < leny; ++i, yt += incy)
        *yt *= beta;
    }
  }

  if (alpha == RealType(0))
    return;

  switch (trans) {
  case 'n':
  case 'N':
    {
      for (int j = 0; j < n; ++j, x += incx) {
        RealType *yt = y;
        const RealType tmp = alpha * (*x);
        for (int i = 0; i < m; ++i, yt += incy)
          *yt += tmp * a[lda*j + i];
      }
    }
    break;
  case 'c':
  case 't':
  case 'C':
  case 'T':
    {
      for (int j = 0; j < n; ++j, y += incy) {
        const RealType *xt = x;
        RealType tmp = RealType(0);
        for (int i = 0; i < m; ++i, xt += incx)
          tmp += (*xt) * a[lda*j + i];

        *y += alpha*tmp;
      }
    }
    break;
  }
}

template<typename RealType>
void ger(int m, int n, const RealType &alpha, const RealType *x, int incx, const RealType *y, int incy, RealType *a, int lda) {
  if (m < 0 || n < 0 || incx == 0 || incy == 0 || lda < std::max(1, m))
    return; // Error

  if (m == 0 || n == 0 || alpha == RealType(0)) 
    return; // Nothing to do

  int jy = 0;
  if (incy < 0)
    jy = -(n-1)*incy;

  if (incx == 1) {
    for (int j = 0; j < n; ++j, jy += incy) {
      if (y[jy] != RealType(0)) {
        const RealType tmp = alpha * y[jy];
        for (int i = 0; i < m; ++i)
          a[lda*j + i] += tmp * x[i];
      }
    }
  }
  else {
    int kx = 0;
    if (incx < 0)
      kx = -(m-1)*incx;

    for (int j = 0; j < n; ++j, jy += incy) {
      if (y[jy] != RealType(0)) {
        const RealType tmp = alpha * y[jy];
        for (int i = 0, ix = kx; i < m; ++i, ix += incx)
          a[lda*j + i] += tmp * x[ix];
      }
    }
  }
}

// Level 3
template<typename RealType>
void gemm(char transa, char transb, int m, int n, int k, const RealType &alpha, const RealType *a, int lda, const RealType *b, int ldb, const RealType &beta, RealType *c, int ldc) {
  if (m < 0 || n < 0 || k < 0 || a == nullptr || b == nullptr || c == nullptr)
    return;

  int nrowa = 0;
  int nrowb = 0;
  unsigned int mode = 0;

  switch (transa) {
  case 'n':
  case 'N':
    nrowa = m;
    //ncola = k;
    mode |= 1;
    break;
  case 'c':
  case 't':
  case 'C':
  case 'T':
    nrowa = k;
    //ncola = m;
    break;
  default:
    std::cerr << "Error: Invalid op '" << transa << "'." << std::endl;
    throw std::runtime_error(std::string("Error: Invalid op '") + transa + "'.");
  }

  switch (transb) {
  case 'n':
  case 'N':
    nrowb = k;
    //ncolb = n;
    mode |= 2;
    break;
  case 'c':
  case 't':
  case 'C':
  case 'T':
    nrowb = n;
    //ncolb = k;
    break;
  default:
    std::cerr << "Error: Invalid op '" << transb << "'." << std::endl;
    throw std::runtime_error(std::string("Error: Invalid op '") + transb + "'.");
  }

  if (lda < std::max(1, nrowa) || ldb < std::max(1, nrowb) || ldc < std::max(1, m))
    return;

  if (m == 0 || n == 0 || ((alpha == RealType(0) || k == 0) && beta == RealType(1)))
    return;

  if (alpha == RealType(0)) {
    if (beta == RealType(0)) {
      for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
          c[ldc*j + i] = RealType(0);
        }
      }
    }
    else if (beta != RealType(1)) {
      for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
          c[ldc*j + i] *= beta;
        }
      }
    }

    return;
  }

  switch (mode) {
  case 0:
    // alpha * A^T * B^T + beta*C
    {
      for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
          RealType tmp = RealType(0);

          for (int l = 0; l < k; ++l)
            tmp += a[lda*i + l]*b[ldb*l + j];

          if (beta == RealType(0))
            c[ldc*j + i] = alpha*tmp;
          else
            c[ldc*j + i] = alpha*tmp + beta*c[ldc*j + i];
        }
      }
    }
    break;
  case 1:
    // alpha * A * B^T + beta*C
    {
      for (int j = 0; j < n; ++j) {
        if (beta == RealType(0)) {
          for (int i = 0; i < m; ++i)
            c[ldc*j + i] = RealType(0);
        }
        else if (beta != RealType(1)) {
          for (int i = 0; i < m; ++i)
            c[ldc*j + i] *= beta;
        }

        for (int l = 0; l < k; ++l) {
          const RealType tmp = alpha*b[ldb*l + j];
          for (int i = 0; i < m; ++i)
            c[ldc*j + i] += tmp*a[lda*l + i];
        }
      }
    }
    break;
  case 2:
    // alpha * A^T * B + beta*C
    {
      for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
          RealType tmp = RealType(0);

          for (int l = 0; l < k; ++l)
            tmp += a[lda*i + l]*b[ldb*j + l];

          if (beta == RealType(0))
            c[ldc*j + i] = alpha*tmp;
          else
            c[ldc*j + i] = alpha*tmp + beta*c[ldc*j + i];
        }
      }
    }
    break;
  case 3:
    // alpha * A*B + beta*C
    {
      for (int j = 0; j < n; ++j) {
        if (beta == RealType(0)) {
          for (int i = 0; i < m; ++i)
            c[ldc*j + i] = RealType(0);
        }
        else if (beta != RealType(1)) {
          for (int i = 0; i < m; ++i)
            c[ldc*j + i] *= beta;
        }

        for (int l = 0; l < k; ++l) {
          const RealType tmp = alpha*b[ldb*j + l];
          for (int i = 0; i < m; ++i)
            c[ldc*j + i] += tmp*a[lda*l + i];
        }
      }
    }
    break;
  }
}

} // end namespace cpu_blas
} // end namespace bleak

#endif // !BLEAK_SLOWBLAS_H
