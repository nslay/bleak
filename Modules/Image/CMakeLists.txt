project(bleakImageModule C CXX)

bleakNewModule(Image SOURCES
  InitializeImageModule.cpp
  Convolution.h Convolution.cpp
  Convolution2D.h Convolution2D.cpp
  MODULE_DEPENDS Common
)

#target_link_libraries(bleakImage PUBLIC bleakCommon) # Even when static, tell bleakExotic about bleakCommon's include directory
#target_include_directories(bleakImage PUBLIC "${bleakImageModule_SOURCE_DIR}")
