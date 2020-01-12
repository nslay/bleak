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
#include "OptimizerFactory.h"
#include "DatabaseFactory.h"

#include "Input.h"
#include "Output.h"
#include "Parameters.h"
#include "CsvReader.h"
#include "DatabaseReader.h"
#include "InnerProduct.h"
#include "Softmax.h"
#include "SoftmaxLoss.h"
#include "L2Loss.h"
#include "HingeLoss.h"
#include "LorenzLoss.h"
#include "Accuracy.h"
#include "Residual.h"
#include "Average.h"
#include "Scale.h"
#include "BatchNormalization.h"
#include "Concatenate.h"
#include "Reshape.h"
#include "ArithmeticOperation.h"

#include "LMDBDatabase.h"

#include "StochasticGradientDescent.h"
#include "AdaGrad.h"
#include "Adam.h"

namespace bleak {

template<typename RealType>
void InitializeCommonModuleTemplate() {
  VertexFactory<RealType> &clVertexFactory = VertexFactory<RealType>::GetInstance();
  OptimizerFactory<RealType> &clOptimizerFactory = OptimizerFactory<RealType>::GetInstance();
  
  clVertexFactory.template Register<Input<RealType>>();
  clVertexFactory.template Register<Output<RealType>>();
  clVertexFactory.template Register<Parameters<RealType>>();
  clVertexFactory.template Register<CsvReader<RealType>>();
  clVertexFactory.template Register<DatabaseReader<RealType>>();
  clVertexFactory.template Register<InnerProduct<RealType>>();
  clVertexFactory.template Register<Softmax<RealType>>();
  clVertexFactory.template Register<SoftmaxLoss<RealType>>();
  clVertexFactory.template Register<L2Loss<RealType>>();
  clVertexFactory.template Register<HingeLoss<RealType>>();
  clVertexFactory.template Register<LorenzLoss<RealType>>();
  clVertexFactory.template Register<Accuracy<RealType>>();
  clVertexFactory.template Register<Residual<RealType>>();
  clVertexFactory.template Register<Average<RealType>>();
  clVertexFactory.template Register<Scale<RealType>>();
  clVertexFactory.template Register<Concatenate<RealType>>();
  clVertexFactory.template Register<Reshape<RealType>>();
  clVertexFactory.template Register<BatchNormalization<RealType>>();
  clVertexFactory.template Register<Plus<RealType>>();
  clVertexFactory.template Register<Minus<RealType>>();
  clVertexFactory.template Register<Multiplies<RealType>>();
  
  clOptimizerFactory.template Register<StochasticGradientDescent<RealType>>();
  clOptimizerFactory.template Register<AdaGrad<RealType>>();
  clOptimizerFactory.template Register<Adam<RealType>>();
}

void InitializeCommonModule() {
  DatabaseFactory &clDBFactory = DatabaseFactory::GetInstance();
  
  clDBFactory.Register<LMDBDatabase>();
  
  InitializeCommonModuleTemplate<float>();
  InitializeCommonModuleTemplate<double>();
}

} // end namespace bleak
