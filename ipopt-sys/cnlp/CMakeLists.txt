cmake_minimum_required( VERSION 3.6 )

project( IpoptCNLP )

set(CMAKE_CXX_FLAGS_DEBUG "-g -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# The following three lines constitute the Ipopt find script
set( Ipopt_DIR "NOTFOUND" CACHE STRING "Ipopt install path")
find_path( Ipopt_INCLUDE_DIR coin/IpIpoptApplication.hpp PATHS ${Ipopt_DIR}/include DOC "Ipopt include directory")
find_library( Ipopt_LIBRARY ipopt PATHS ${Ipopt_DIR}/lib DOC "Ipopt library")

set( library_name ipopt_cnlp )

set( CAPI_HEADERS src/c_api.h )

if (APPLE)
  # NOTE: The following two lines use @rpath and install an effective install path to LC_RPATH in the
  # dylib
  #set(CMAKE_SKIP_INSTALL_RPATH FALSE)
  #set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

  # Since there are some glitches with reexporting cdylibs within rust for libraries that depend on
  # other dylibs, we use explicit install names here so that they can be tracked easily.
  # The downside of this is that now libipopt_cnlp.dylib is largely immobile, however this can be
  # changed manually with install_name_tool.
  set( CMAKE_INSTALL_NAME_DIR "${CMAKE_INSTALL_PREFIX}/lib")
endif(APPLE)

# Add a library and its source files.
add_library( ${library_name} STATIC
    ${CAPI_HEADERS} 
    src/nlp.hpp
    src/c_api.cpp
    src/nlp.cpp
)

add_definitions(-DHAVE_CSTDDEF)

target_link_libraries( ${library_name} PUBLIC ${Ipopt_LIBRARY} ${LINK_LIBS})

target_include_directories( ${library_name} PRIVATE ${Ipopt_INCLUDE_DIR} )

install( TARGETS ${library_name} DESTINATION lib )
install( FILES ${CAPI_HEADERS} DESTINATION include )
