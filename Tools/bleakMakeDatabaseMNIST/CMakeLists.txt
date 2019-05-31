project(bleakMakeDatabaseMNIST C CXX)

add_executable(bleakMakeDatabaseMNIST bleakMakeDatabaseMNIST.cpp bsdgetopt.h bsdgetopt.c)
target_link_libraries(bleakMakeDatabaseMNIST ${bleakLibraries})
