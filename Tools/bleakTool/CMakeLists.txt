project(bleakTool C CXX)

add_executable(bleakTool bleakTool.cpp bsdgetopt.h bsdgetopt.c)
target_link_libraries(bleakTool ${bleakLibraries})
