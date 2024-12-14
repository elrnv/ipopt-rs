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

if(MAKE_AND_RUN_TEST)
    set( Ipopt_LIBRARY_DIRS "NOTFOUND" CACHE STRING "Ipopt library paths")
    find_path( Ipopt_LIBRARY_DIR libipopt.a PATHS ${Ipopt_LIBRARY_DIRS} DOC "Ipopt library directory")


    # Add the test executable and its source files.
    add_executable( hs071 
        ${CAPI_HEADERS} 
        tests/hs071_nlp.h
        tests/hs071_nlp.c
        tests/hs071_main.c
    )

    set( MKLROOT /home/egor )

    target_link_directories( hs071 PRIVATE ${MKLROOT}/lib )

    target_link_libraries( hs071 PRIVATE
        ${library_name}
        ${Ipopt_LIBRARY_DIR}/libipopt.a
        ${MKLROOT}/lib/libmkl_intel_lp64.a
        ${MKLROOT}/lib/libmkl_tbb_thread.a
        ${MKLROOT}/lib/libmkl_core.a
        ${MKLROOT}/lib/libmkl_intel_lp64.a
        ${MKLROOT}/lib/libmkl_tbb_thread.a
        ${MKLROOT}/lib/libmkl_core.a
        ${MKLROOT}/lib/libmkl_intel_lp64.a
        ${MKLROOT}/lib/libmkl_tbb_thread.a
        ${MKLROOT}/lib/libmkl_core.a
         tbb stdc++ pthread m dl
        )

    add_custom_command(TARGET hs071 POST_BUILD
        COMMENT "Run hs071 test"
        COMMAND "$<TARGET_FILE:hs071>"
    )
endif(MAKE_AND_RUN_TEST)
