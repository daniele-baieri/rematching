cmake_minimum_required(VERSION 3.16.0)
project(ReMatching LANGUAGES C CXX)


# Build and install libCUT
include(ExternalProject)
ExternalProject_Add(CUT
                    SOURCE_DIR "${CMAKE_SOURCE_DIR}/ext/cut"
                    BINARY_DIR "${CMAKE_SOURCE_DIR}/build-cut"
                    INSTALL_DIR "${CMAKE_SOURCE_DIR}/install"
                    CMAKE_ARGS "-DCMAKE_INSTALL_PREFIX=${CMAKE_SOURCE_DIR}/install")

# Include dependencies
include_directories("${CMAKE_SOURCE_DIR}/ext/eigen")
include_directories("${CMAKE_SOURCE_DIR}/ext/libigl/include")
include_directories("${CMAKE_SOURCE_DIR}/ext/json/include")
include_directories("${CMAKE_SOURCE_DIR}/install/cut/include")
if(MSVC)
    set(libCUT "${CMAKE_SOURCE_DIR}/install/cut/lib/cut.lib")
elseif(UNIX)
    set(libCUT "${CMAKE_SOURCE_DIR}/install/cut/lib/libcut.a")
endif()
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)


# Add Polyscope to submodules to compile
add_subdirectory("${CMAKE_SOURCE_DIR}/ext/polyscope")


# Include rematching library
include_directories("${CMAKE_SOURCE_DIR}/include")
add_library(RMT STATIC  "${CMAKE_SOURCE_DIR}/src/rmt/graph.cpp"
                        "${CMAKE_SOURCE_DIR}/src/rmt/mesh.cpp"
                        "${CMAKE_SOURCE_DIR}/src/rmt/voronoifps.cpp"
                        "${CMAKE_SOURCE_DIR}/src/rmt/region.cpp"
                        "${CMAKE_SOURCE_DIR}/src/rmt/regiondict.cpp"
                        "${CMAKE_SOURCE_DIR}/src/rmt/flatunion.cpp"
                        "${CMAKE_SOURCE_DIR}/src/rmt/reconstruction.cpp"
                        "${CMAKE_SOURCE_DIR}/src/rmt/preprocess.cpp"
                        "${CMAKE_SOURCE_DIR}/src/rmt/clean.cpp"
                        "${CMAKE_SOURCE_DIR}/src/rmt/weightmap.cpp"
                        "${CMAKE_SOURCE_DIR}/src/rmt/eval.cpp"
                        "${CMAKE_SOURCE_DIR}/src/rmt/io.cpp"
                        "${CMAKE_SOURCE_DIR}/src/rmt/rmt.cpp")
target_link_libraries(RMT ${libCUT} Threads::Threads)
set_target_properties(RMT PROPERTIES CXX_STANDARD 17)


# Installation settings
install(DIRECTORY "${CMAKE_SOURCE_DIR}/include/rmt"
        DESTINATION "${CMAKE_INSTALL_PREFIX}/rmt/include")
install(TARGETS RMT
        LIBRARY DESTINATION "${CMAKE_INSTALL_PREFIX}/rmt/lib"
        ARCHIVE DESTINATION "${CMAKE_INSTALL_PREFIX}/rmt/lib"
        RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/rmt/bin")



# Remeshing demo
add_executable(Remesh "${CMAKE_SOURCE_DIR}/src/apps/remesh.cpp")
target_link_libraries(Remesh RMT)
set_target_properties(Remesh PROPERTIES CXX_STANDARD 17)

# Remeshing batch
add_executable(BatchRemesh "${CMAKE_SOURCE_DIR}/src/apps/batch.cpp")
target_link_libraries(BatchRemesh RMT)
set_target_properties(BatchRemesh PROPERTIES CXX_STANDARD 17)

# Geodesic Voronoi decomposition
add_executable(Decomp "${CMAKE_SOURCE_DIR}/src/apps/decomp.cpp")
target_link_libraries(Decomp RMT)
target_link_libraries(Decomp polyscope)
set_target_properties(Decomp PROPERTIES CXX_STANDARD 17)