/*-
 * Copyright (c) 2018 Nathan Lay (enslay@gmail.com)
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

#include "ImageToMatrix.h"

namespace bleak {

template class ImageToMatrix<float, 1>;
template class ImageToMatrix<double, 1>;

template class ImageToMatrix<float, 2>;
template class ImageToMatrix<double, 2>;

template class ImageToMatrix<float, 3>;
template class ImageToMatrix<double, 3>;

} // end namespace bleak

#if 0
// Test code...
#include <iostream>
#include <vector>

int main(int argc, char **argv) {
  bleak::ImageToMatrix<float, 2> clIm2Col;

  //
  // Reference 2D access pattern for a 6x5 image with padding
  //
  // -1 -1 -1 -1 -1 -1 -1
  // -1  0  1  2  3  4 -1
  // -1  5  6  7  8  9 -1
  // -1 10 11 12 13 14 -1
  // -1 15 16 17 18 19 -1
  // -1 20 21 22 23 24 -1
  // -1 25 26 27 28 29 -1
  // -1 -1 -1 -1 -1 -1 -1
  //

  clIm2Col.kernelSize[0] = 3;
  clIm2Col.kernelSize[1] = 4;

  clIm2Col.stride[0] = 1;
  clIm2Col.stride[1] = 1;

  clIm2Col.dilate[0] = 1;
  clIm2Col.dilate[1] = 1;

  clIm2Col.padding[0] = 1;
  clIm2Col.padding[1] = 1;

  const int a_iImageSize[3] = { 3, 6, 5 }; // 3 channels, 6 rows, 5 columns

  if (!clIm2Col.Good(a_iImageSize)) {
    std::cerr << "Error: Bad image size." << std::endl;
    return -1;
  }

  int rows = 0, cols = 0;
  clIm2Col.ComputeMatrixDimensions(rows, cols, a_iImageSize);

  std::vector<int> vIndexMatrix(rows*cols, 0);

  clIm2Col.ExtractIndexMatrix(vIndexMatrix.data(), a_iImageSize);

  std::cout << "Index matrix: " << std::endl;

  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j)
      std::cout << vIndexMatrix[cols*i + j] << ' ';

    std::cout << std::endl;
  }

  return 0;
}
#endif 
