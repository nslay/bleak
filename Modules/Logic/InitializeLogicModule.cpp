#include "VertexFactory.h"
#include "MaskParameters.h"
#include "LogicOperation.h"
#include "LogicInnerProduct.h"
#include "LeftShift.h"
#include "PlusXorLoss.h"

namespace bleak {

template<typename RealType>
void InitializeLogicModuleTemplate() {
  VertexFactory<RealType> &clVertexFactory = VertexFactory<RealType>::GetInstance();

  clVertexFactory.template Register<MaskParameters<RealType>>();
  clVertexFactory.template Register<PlusOr<RealType>>();
  clVertexFactory.template Register<PlusAnd<RealType>>();
  clVertexFactory.template Register<PlusXor<RealType>>();
  clVertexFactory.template Register<InnerProductPlusOr<RealType>>();
  clVertexFactory.template Register<InnerProductPlusAnd<RealType>>();
  clVertexFactory.template Register<LeftShift<RealType>>();
  clVertexFactory.template Register<PlusXorLoss<RealType>>();
}

void InitializeLogicModule() {
  InitializeLogicModuleTemplate<float>();
  InitializeLogicModuleTemplate<double>();
}

} // end namespace bleak
