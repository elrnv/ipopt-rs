cmake_minimum_required( VERSION 3.6 )

project( IpoptCNLP )

set(CMAKE_CXX_FLAGS_DEBUG "-g -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_STANDARD 11)

# The following three lines constitute the Ipopt find script
set( Ipopt_INCLUDE_DIRS "NOTFOUND" CACHE STRING "Ipopt include paths")
find_path( Ipopt_INCLUDE_DIR coin/IpIpoptApplication.hpp PATHS ${Ipopt_INCLUDE_DIRS} DOC "Ipopt include directory")

set( library_name ipopt_cnlp )

set( CAPI_HEADERS src/c_api.h )

# Add a library and its source files.
add_library( ${library_name} STATIC
    ${CAPI_HEADERS} 
    src/nlp.hpp
    src/c_api.cpp
    src/nlp.cpp
)

add_definitions(-DHAVE_CSTDDEF)

target_include_directories( ${library_name} PRIVATE ${Ipopt_INCLUDE_DIR} )

install( TARGETS ${library_name} DESTINATION lib )
install( FILES ${CAPI_HEADERS} DESTINATION include )
