project(bleakProstateXTool C CXX)

# If bleakITK is enabled, then we can probably safely build this tool
if (bleakITK_ENABLE)
  add_executable(bleakProstateXTool bleakProstateXTool.cpp ProstateXCommon.h ProstateXCommon.cpp bsdgetopt.h bsdgetopt.c)
  target_link_libraries(bleakProstateXTool ${bleakLibraries})
  
  add_executable(bleakEvaluateProstateX bleakEvaluateProstateX.cpp ProstateXCommon.h ProstateXCommon.cpp bsdgetopt.h bsdgetopt.c)
  target_link_libraries(bleakEvaluateProstateX ${bleakLibraries})
endif()
