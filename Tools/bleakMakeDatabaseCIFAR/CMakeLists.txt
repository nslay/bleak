project(bleakMakeDatabaseCIFAR C CXX)

add_executable(bleakMakeDatabaseCIFAR bleakMakeDatabaseCIFAR.cpp bsdgetopt.h bsdgetopt.c)
target_link_libraries(bleakMakeDatabaseCIFAR ${bleakLibraries})

if (bleakITK_ENABLE)
  target_compile_definitions(bleakMakeDatabaseCIFAR PRIVATE -DWITH_ITK=1)
endif()
