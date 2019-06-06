project(bleakTreesModule C CXX)

bleakNewModule(Trees SOURCES
  InitializeTreesModule.cpp
  FeatureSelectionAnnealing.h FeatureSelectionAnnealing.cpp
  RandomHingeFerns.h RandomHingeFerns.cpp
  RandomHingeForest.h RandomHingeForest.cpp
  TreeLorenzLoss.h TreeLorenzLoss.cpp
  TreeL2Regularization.h TreeL2Regularization.cpp
  MODULE_DEPENDS Common
)

#target_link_libraries(bleakTrees PUBLIC bleakCommon) # Even when static, tell bleakTrees about bleakCommon's include directory
#target_include_directories(bleakTrees PUBLIC "${bleakTreesModule_SOURCE_DIR}")
