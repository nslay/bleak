cmake_minimum_required(VERSION 3.10)

project(bleak C CXX)

# All modules that are available
set(bleakModules "" CACHE INTERNAL "Modules")

# All libraries to link with (include information is propagated)
set(bleakLibraries "" CACHE INTERNAL "Libraries")

include("cmake/bleakModuleMacros.cmake")
include("CheckCSourceCompiles")
include("CheckLanguage")

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wno-unused-value")
elseif (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
elseif (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
  add_definitions(-DYY_NO_UNISTD_H=1 -D_USE_MATH_DEFINES=1)
endif()

set(bleakUseOpenMP OFF CACHE BOOL "Compile with OpenMP support (if available).")
set(bleakUseCUDA OFF CACHE BOOL "Compile with CUDA support (if available).")

if (bleakUseOpenMP)
  find_package(OpenMP)
  
  if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  else()
    message(WARNING "OpenMP not found. Disabling.")
    set(bleakUseOpenMP OFF CACHE BOOL "" FORCE)
  endif()
endif()

if (bleakUseCUDA)
  check_language(CUDA)
  
  if (CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    add_definitions(-DBLEAK_USE_CUDA=1)
    
    # Disable annoying warnings in nvcc
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe --diag_suppress=expr_has_no_effect -Xcudafe --diag_suppress=set_but_not_used -Xcudafe --diag_suppress=declared_but_not_referenced")
  else()
    message(WARNING "CUDA not found. Disabling.")
    set(bleakUseCUDA OFF CACHE BOOL "" FORCE)
  endif()
endif()

add_library(lmdb
  lmdb/libraries/liblmdb/lmdb.h   
  lmdb/libraries/liblmdb/mdb.c
  lmdb/libraries/liblmdb/midl.h lmdb/libraries/liblmdb/midl.c
)
target_include_directories(lmdb PUBLIC lmdb/libraries/liblmdb)

if (WIN32)
  target_link_libraries(lmdb PUBLIC ntdll)
elseif (UNIX)
  target_link_libraries(lmdb PUBLIC pthread)

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


