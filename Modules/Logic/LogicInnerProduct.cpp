#include "LogicInnerProduct.h"

namespace bleak {

#define INSTANTIATE_INNERPRODUCT_OPERATION(vertexName, opType, realType) \
template struct opType < realType >; \
template class LogicInnerProduct< realType , opType >; \
template class vertexName < realType >

INSTANTIATE_INNERPRODUCT_OPERATION(InnerProductPlusOr, PlusOrOp, float);
INSTANTIATE_INNERPRODUCT_OPERATION(InnerProductPlusOr, PlusOrOp, double);
INSTANTIATE_INNERPRODUCT_OPERATION(InnerProductPlusAnd, PlusAndOp, float);
INSTANTIATE_INNERPRODUCT_OPERATION(InnerProductPlusAnd, PlusAndOp, double);

} // end namespace bleak
