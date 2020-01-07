project(bleakImageModule C CXX)

bleakNewModule(Image SOURCES
  InitializeImageModule.cpp
  Convolution.h Convolution.cpp
  ConvolutionNaive.h ConvolutionNaive.cpp
  ImageToMatrix.h ImageToMatrix.cpp
  MODULE_DEPENDS Common
)

#target_link_libraries(bleakImage PUBLIC bleakCommon) # Even when static, tell bleakExotic about bleakCommon's include directory
#target_include_directories(bleakImage PUBLIC "${bleakImageModule_SOURCE_DIR}")

#add_executable(main ImageToMatrix.cpp)
#target_link_libraries(main bleakCommon bleakImage)
