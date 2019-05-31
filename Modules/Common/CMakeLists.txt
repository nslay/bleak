project(bleakCommonModule C CXX)

bleakNewModule(Common SOURCES
Parser/y.tab.h Parser/y.tab.c
Parser/lex.yy.h Parser/lex.yy.c
Parser/bleak_graph_ast.h Parser/bleak_graph_ast.c
Parser/bleak_vector.h Parser/bleak_vector.cpp
Parser/bleak_expression.h Parser/bleak_expression.cpp
Parser/bleak_parser.h Parser/bleak_parser.c
SadLoader.h SadLoader.cpp
Subgraph.h Subgraph.cpp

InitializeModules.h InitializeModules.cpp
Size.h
Common.h Common.cpp
Timer.h Timer.cpp
Array.h Array.cpp
Edge.h Edge.cpp
Vertex.h Vertex.cpp
Factory.h
VertexFactory.h VertexFactory.cpp
ParameterContainer.h
Property.h Property.cpp
IniFile.h IniFile.cpp
Graph.h Graph.cpp
Input.h Input.cpp
Parameters.h Parameters.cpp
InnerProduct.h InnerProduct.cpp
Scale.h Scale.cpp
Reshape.h Reshape.cpp
BatchNormalization.h BatchNormalization.cpp
Softmax.h Softmax.cpp
SoftmaxLoss.h SoftmaxLoss.cpp
L2Loss.h L2Loss.cpp
HingeLoss.h HingeLoss.cpp
LorenzLoss.h LorenzLoss.cpp
CsvReader.h CsvReader.cpp
DatabaseReader.h DatabaseReader.cpp
PrintOutput.h
Accuracy.h Accuracy.cpp
Residual.h Residual.cpp
Average.h Average.cpp
Concatenate.h Concatenate.cpp
ArithmeticOperation.h ArithmeticOperation.cpp
Optimizer.h Optimizer.cpp
IterativeOptimizer.h IterativeOptimizer.cpp
OptimizerFactory.h OptimizerFactory.cpp
StochasticGradientDescent.h StochasticGradientDescent.cpp
AdaGrad.h AdaGrad.cpp
Adam.h Adam.cpp
Database.h
DatabaseFactory.h DatabaseFactory.cpp
LMDBDatabase.h LMDBDatabase.cpp

InitializeCommonModule.cpp
LIBRARIES lmdb
ENABLE_BY_DEFAULT
)

#target_link_libraries(bleakCommon PUBLIC lmdb ${PLATFORM_LIBS}) # Even when static, tell bleakCommon about lmdb's include directory
#target_include_directories(bleakCommon PUBLIC "${bleakCommonModule_SOURCE_DIR}")
