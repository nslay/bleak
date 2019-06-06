project(bleakExoticModule C CXX)

bleakNewModule(Exotic SOURCES
  InitializeExoticModule.cpp
  HellingerLoss.h HellingerLoss.cpp
  SignedLikelihoodAccuracy.h SignedLikelihoodAccuracy.cpp
  MODULE_DEPENDS Common
)

#target_link_libraries(bleakExotic PUBLIC bleakCommon) # Even when static, tell bleakExotic about bleakCommon's include directory
#target_include_directories(bleakExotic PUBLIC "${bleakExoticModule_SOURCE_DIR}")
