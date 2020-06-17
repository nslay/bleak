project(bleakCudnn C CXX)

if (bleakCudnn_ENABLE)
  if (NOT bleakUseCUDA)
    message(FATAL_ERROR "CUDA is required to build this module.")
  endif()
  
  # Try to deduce CUDA toolkit root folder
  get_filename_component(CUDA_ROOT_DIR "${CMAKE_CUDA_COMPILER}" DIRECTORY)
  get_filename_component(CUDA_ROOT_DIR "${CUDA_ROOT_DIR}" DIRECTORY)
  
  set(bleakCudnnIncludeDir "${CUDA_ROOT_DIR}/include" CACHE PATH "CUDNN include directory.")
  set(bleakCudnnLibraryDir "${CUDA_ROOT_DIR}/lib" CACHE PATH "CUDNN library directory.")
  
  find_library(bleakCudnnLibrary cudnn "${bleakCudnnLibraryDir}")

  if (NOT bleakCudnnLibrary)
    message(FATAL_ERROR "Could not find cudnn library.")
  endif()
endif()

bleakNewModule(Cudnn SOURCES
  InitializeCudnnModule.cpp
  CudnnCommon.h CudnnCommon.cpp
  CudnnConvolution.h CudnnConvolution.cpp
  INCLUDE_DIRECTORIES "${bleakCudnnIncludeDir}"
  MODULE_DEPENDS Common
  LIBRARIES "${bleakCudnnLibrary}"
)
