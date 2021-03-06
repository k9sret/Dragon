message(STATUS "Found CXX Module: ${CMAKE_CURRENT_LIST_DIR}")

FILE(GLOB_RECURSE MODULE_FILES *.h *.hpp *.c *.cpp *.cu *.cc)
FILE(GLOB_RECURSE SRC_FILES ../../src/*.c ../../src/*.cpp ../../src/*.cu ../../src/*.cc)
LIST(REMOVE_ITEM SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/../../src/operators/misc/python_op.cc)

# ---[ Target
if (WITH_CUDA)
    CUDA_ADD_LIBRARY(${PROJECT_NAME}_cxx SHARED ${MODULE_FILES} ${SRC_FILES})
    TARGET_LINK_LIBRARIES(${PROJECT_NAME}_cxx  ${CUDA_LIBRARIES} ${CUDA_cublas_LIBRARY} ${CUDA_curand_LIBRARY})
else ()
    ADD_LIBRARY(${PROJECT_NAME}_cxx SHARED ${MODULE_FILES} ${SRC_FILES})
endif()

# ---[ Linker(Basics)
FILE(GLOB targets ${3RDPARTY_LIBS}/*.so ${3RDPARTY_LIBS}/*.lib) 
foreach(target ${targets})
    TARGET_LINK_LIBRARIES(${PROJECT_NAME}_cxx  ${target})
endforeach()

# ---[ Linker(OptionalLibs)
if (UNIX AND WITH_CUDNN)
    TARGET_LINK_LIBRARIES(${PROJECT_NAME}_cxx cudnn)
endif()
if (UNIX AND WITH_BLAS)
    TARGET_LINK_LIBRARIES(${PROJECT_NAME}_cxx openblas)
endif()
if (UNIX AND WITH_MPI_NCCL)
    TARGET_LINK_LIBRARIES(${PROJECT_NAME}_cxx nccl)
endif()

# ---[ Linker(Platforms)
if(UNIX)
    TARGET_LINK_LIBRARIES(${PROJECT_NAME}_cxx protobuf pthread)
endif()
if(WIN32)
    TARGET_LINK_LIBRARIES(${PROJECT_NAME}_cxx shlwapi.lib)
endif()

SET_TARGET_PROPERTIES(${PROJECT_NAME}_cxx PROPERTIES OUTPUT_NAME dragon_cxx)

# ---[ Install
INSTALL(TARGETS ${PROJECT_NAME}_cxx DESTINATION ${PROJECT_BINARY_DIR}/../lib)