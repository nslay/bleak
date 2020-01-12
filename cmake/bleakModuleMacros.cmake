macro(bleakNewModule name)
  if ("${name}" IN_LIST "${bleakModules}")
    MESSAGE(FATAL_ERROR "Bleak module ${name} already exists.")
  endif()
  
  set(optionKeyWords ENABLE_BY_DEFAULT)
  set(oneValueKeyWords "")
  set(multiValueKeyWords MODULE_DEPENDS LIBRARIES SOURCES INCLUDE_DIRECTORIES DEFINITIONS)
  
  cmake_parse_arguments("bleak${name}" "${optionKeyWords}" "${oneValueKeyWords}" "${multiValueKeyWords}" ${ARGN} )
  
  set(defaultEnable OFF)
  
  if (bleak${name}_ENABLE_BY_DEFAULT)
    set(defaultEnable ON)
  endif()
  
  set("bleak${name}_ENABLE" ${defaultEnable} CACHE BOOL "Enable ${name} module.")
  
  # Remove leading/trailing white space
  string(REGEX REPLACE "^[ \t\r\n]+" "" bleak${name}_SOURCES "${bleak${name}_SOURCES}")
  string(REGEX REPLACE "[ \t\r\n]+$" "" bleak${name}_SOURCES "${bleak${name}_SOURCES}")
  
  if (bleak${name}_ENABLE)
    add_library("bleak${name}" ${bleak${name}_SOURCES})
    
    set(dependencies "")
    
    foreach (dependency ${bleak${name}_MODULE_DEPENDS})
      set(dependencies "bleak${dependency}" ${dependencies})
    endforeach()
    
    set(dependencies ${dependencies} ${bleak${name}_LIBRARIES})
    
    # Remove leading/trailing white space
    string(REGEX REPLACE "^[ \t\r\n]+" "" dependencies "${dependencies}")
    string(REGEX REPLACE "[ \t\r\n]+$" "" dependencies "${dependencies}")
    
    target_link_libraries("bleak${name}" PUBLIC ${dependencies})
    target_include_directories("bleak${name}" PUBLIC "${PROJECT_SOURCE_DIR}" ${bleak${name}_INCLUDE_DIRECTORIES})
    target_compile_definitions("bleak${name}" PUBLIC ${bleak${name}_DEFINITIONS})
  endif()
  
  list(APPEND bleakModules "${name}")
  set(bleakModules "${bleakModules}" PARENT_SCOPE)
endmacro()

macro(bleakGenerateInitializeModules)
  set(BLEAK_DECLARE_INITIALIZE_FUNCTIONS "")
  set(BLEAK_CALL_INITIALIZE_FUNCTIONS "")
  
  foreach (module ${bleakModules})
    if (NOT bleak${module}_ENABLE)
      continue()
    endif()
    
    set(BLEAK_DECLARE_INITIALIZE_FUNCTIONS "${BLEAK_DECLARE_INITIALIZE_FUNCTIONS}\nvoid Initialize${module}Module();")
    set(BLEAK_CALL_INITIALIZE_FUNCTIONS "  ${BLEAK_CALL_INITIALIZE_FUNCTIONS}\n  Initialize${module}Module();")
  endforeach()
  
  set(initRootFolder "${bleak_SOURCE_DIR}/Modules/Common")
  
  configure_file("${initRootFolder}/InitializeModules.cpp.in" "${initRootFolder}/InitializeModules.cpp")

endmacro()

macro(bleakCheckDependencies)
  foreach(module ${bleakModules})
    if (NOT TARGET bleak${module})
      continue()
    endif()
  
    get_target_property(dependencies "bleak${module}" INTERFACE_LINK_LIBRARIES)
  
    foreach (dependency ${dependencies})
      string(REGEX MATCH "^bleak" result "${dependency}")
      string(COMPARE NOTEQUAL "bleak" "${result}" result)
      
      if (result)
        continue()
      endif()
      
      set("${dependency}_ENABLE" ON CACHE BOOL "" FORCE)
    endforeach()
  endforeach()
endmacro()

macro(bleakGetAllLibraries libraries)
  set(${libraries} "")
  
  foreach(module ${bleakModules})
    #if (bleak${module}_ENABLE)
    if (TARGET bleak${module})
      set(${libraries} ${${libraries}} bleak${module})
    endif()
  endforeach()
  
endmacro()

