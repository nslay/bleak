project(bleakMakeDatabaseCIFAR C CXX)

add_executable(bleakMakeDatabaseCIFAR bleakMakeDatabaseCIFAR.cpp bsdgetopt.h bsdgetopt.c)
target_link_libraries(bleakMakeDatabaseCIFAR ${bleakLibraries})
