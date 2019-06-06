/*-
 * Copyright (c) 2017 Nathan Lay (enslay@gmail.com)
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

#include "VertexFactory.h"

#include "FeatureSelectionAnnealing.h"
#include "RandomHingeFerns.h"
#include "RandomHingeForest.h"
#include "TreeLorenzLoss.h"
#include "TreeL2Regularization.h"

namespace bleak {

template<typename RealType>
void InitializeTreesModuleTemplate() {
  VertexFactory<RealType> &clVertexFactory = VertexFactory<RealType>::GetInstance();

  clVertexFactory.template Register<FeatureSelectionAnnealing<RealType>>();
  clVertexFactory.template Register<RandomHingeFerns<RealType>>();
  clVertexFactory.template Register<RandomHingeForest<RealType>>();
  clVertexFactory.template Register<TreeLorenzLoss<RealType>>();
  clVertexFactory.template Register<TreeL2Regularization<RealType>>();
}

void InitializeTreesModule() {
  InitializeTreesModuleTemplate<float>();
  InitializeTreesModuleTemplate<double>();
}

} // end namespace bleak
