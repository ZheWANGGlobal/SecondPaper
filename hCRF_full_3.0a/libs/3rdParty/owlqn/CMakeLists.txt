cmake_minimum_required(VERSION 2.6)
project(owlqn)
IF(UNIX)
  IF(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

  ENDIF()
ENDIF()

INCLUDE_DIRECTORIES(${owlqn_SOURCE_DIR}/include)

SET(LIB_SRC src/leastSquares.cpp src/logreg.cpp src/main.cpp src/OWLQN.cpp src/TerminationCriterion.cpp)
add_library(owlqn ${LIB_SRC})

SET(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR})
INSTALL_TARGETS(/lib owlqn)
