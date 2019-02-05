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

set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

# Add a library and its source files.
add_library( ${library_name} SHARED
    ${CAPI_HEADERS} 
    src/nlp.hpp
    src/c_api.cpp
    src/nlp.cpp
)

# Link against the Houdini libraries, and add required include directories and
# compile definitions.
target_link_libraries( ${library_name} ${Ipopt_LIBRARY} )

target_include_directories( ${library_name} PRIVATE ${Ipopt_INCLUDE_DIR} )

install( TARGETS ${library_name} DESTINATION lib )
install( FILES ${CAPI_HEADERS} DESTINATION include )