#include "LogicOperation.h"

namespace bleak {

#define INSTANTIATE_BINARY_OPERATION(vertexName, opType, realType) \
template struct opType < realType >; \
template class BinaryOperation< realType , opType >; \
template class vertexName< realType >

INSTANTIATE_BINARY_OPERATION(PlusOr, PlusOrOp, float);
INSTANTIATE_BINARY_OPERATION(PlusOr, PlusOrOp, double);
INSTANTIATE_BINARY_OPERATION(PlusAnd, PlusAndOp, float);
INSTANTIATE_BINARY_OPERATION(PlusAnd, PlusAndOp, double);
INSTANTIATE_BINARY_OPERATION(PlusXor, PlusXorOp, float);
INSTANTIATE_BINARY_OPERATION(PlusXor, PlusXorOp, double);

} // end namespace bleak
