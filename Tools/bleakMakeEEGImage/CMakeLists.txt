project(bleakMakeEEGImage C CXX)

# If bleakITK is enabled, then we can probably safely build this tool
if (bleakITK_ENABLE)
  add_executable(bleakMakeEEGImage bleakMakeEEGImage.cpp bsdgetopt.h bsdgetopt.c)
  target_link_libraries(bleakMakeEEGImage ${bleakLibraries})
endif()
