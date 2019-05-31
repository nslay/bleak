project(bleakSadToDot C CXX)

add_executable(bleakSadToDot bleakSadToDot.cpp bsdgetopt.h bsdgetopt.c)
target_link_libraries(bleakSadToDot ${bleakLibraries})
