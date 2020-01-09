project(bleakITKModule C CXX)

if (bleakITK_ENABLE)
  find_package(ITK 4)

  if (NOT ITK_FOUND)
    message(FATAL_ERROR "The ITK library is required to build this module.")
  endif()
  
  include(${ITK_USE_FILE})
endif()

bleakNewModule(ITK SOURCES
  InitializeITKModule.cpp
  ITKImageLoader.h ITKImageLoader.cpp
  MODULE_DEPENDS Common
  LIBRARIES ${ITK_LIBRARIES}
)
