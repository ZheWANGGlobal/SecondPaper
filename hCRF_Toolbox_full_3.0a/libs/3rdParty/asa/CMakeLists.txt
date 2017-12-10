cmake_minimum_required(VERSION 2.6)
project(ASA)

INCLUDE_DIRECTORIES(${ASA_SOURCE_DIR}/include)
add_library(asa src/asa.c)
SET(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR})
INSTALL_TARGETS(/lib asa)
