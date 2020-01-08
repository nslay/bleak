cmake_minimum_required(VERSION 3.10)

project(bleak C CXX)

# All modules that are available
set(bleakModules "" CACHE INTERNAL "Modules")

# All libraries to link with (include information is propagated)
set(bleakLibraries "" CACHE INTERNAL "Libraries")

include("cmake/bleakModuleMacros.cmake")
include("CheckCSourceCompiles")

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wno-unused-value")
elseif (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
elseif (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
  add_definitions(-DYY_NO_UNISTD_H=1 -D_USE_MATH_DEFINES=1)
endif()

if (WIN32)
  set(PLATFORM_LIBS ntdll)
elseif(UNIX)
  set(PLATFORM_LIBS pthread)
else()
  set(PLATFORM_LIBS )
endif()

# NOTE: Needs ntdll on Windows
add_library(lmdb
  lmdb/libraries/liblmdb/lmdb.h   
  lmdb/libraries/liblmdb/mdb.c
  lmdb/libraries/liblmdb/midl.h lmdb/libraries/liblmdb/midl.c
)
target_link_libraries(lmdb PUBLIC ${PLATFORM_LIBS})
target_include_directories(lmdb PUBLIC lmdb/libraries/liblmdb)

if (UNIX)
  check_c_source_compiles(
  "
#include <sys/sem.h>
int main(int argc, char **argv) { 
#ifndef _SEM_SEMUN_UNDEFINED
  union semun semu;
#endif
  return 0;
}
  "
  HAS_SEMUN_OR_DEF)

  if (NOT HAS_SEMUN_OR_DEF)
    target_compile_definitions(lmdb PRIVATE -D_SEM_SEMUN_UNDEFINED=1)
  endif()
endif()
  
file(GLOB moduleFolders Modules/*)
foreach (moduleFolder ${moduleFolders})
  if (IS_DIRECTORY "${moduleFolder}")
    add_subdirectory("${moduleFolder}")
  endif()
endforeach()

bleakCheckDependencies()
bleakGenerateInitializeModules()
bleakGetAllLibraries(bleakLibraries)

file(GLOB toolFolders Tools/*)
foreach (toolFolder ${toolFolders})
  if (IS_DIRECTORY "${toolFolder}")
    add_subdirectory("${toolFolder}")
  endif()
endforeach()

file(GLOB exampleFolders Examples/*)
foreach (exampleFolder ${exampleFolders})
  if (IS_DIRECTORY "${exampleFolder}")
    add_subdirectory("${exampleFolder}")
  endif()
endforeach()


