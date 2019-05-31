project(bleakMakeDatabase C CXX)

add_executable(bleakMakeDatabase bleakMakeDatabase.cpp bsdgetopt.h bsdgetopt.c)
target_link_libraries(bleakMakeDatabase ${bleakLibraries})
