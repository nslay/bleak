project(bleakProstateXTool C CXX)

# If bleakITK is enabled, then we can probably safely build this tool
if (bleakITK_ENABLE)
  add_executable(bleakProstateXTool bleakProstateXTool.cpp bsdgetopt.h bsdgetopt.c)
  target_link_libraries(bleakProstateXTool ${bleakLibraries})
endif()
