project(bleakLogicModule C CXX)

bleakNewModule(Logic SOURCES
  InitializeLogicModule.cpp
  LogicCommon.h
  MaskParameters.h MaskParameters.cpp
  LogicOperation.h LogicOperation.cpp
  LogicInnerProduct.h LogicInnerProduct.cpp
  LeftShift.h LeftShift.cpp
  PlusXorLoss.h PlusXorLoss.cpp
  MODULE_DEPENDS Common
)

#target_link_libraries(bleakImage PUBLIC bleakCommon) # Even when static, tell bleakExotic about bleakCommon's include directory
#target_include_directories(bleakImage PUBLIC "${bleakImageModule_SOURCE_DIR}")

#add_executable(main main.cpp)
#target_link_libraries(main bleakCommon bleakLogic)
