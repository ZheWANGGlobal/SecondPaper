cmake_minimum_required(VERSION 2.6)
project(hCRF)

ADD_DEFINITIONS(-DUSEOWL -DUSELBFGS)

# We want -fPIC on linux, to be able to link to dynamic library
# IF(${UNIX})
#   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
# ENDIF()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra ")

# For debugging. We also want the extra warning of Effective c++
IF (CMAKE_BUILD_TYPE STREQUAL "Debug")
  ADD_DEFINITIONS(-D_DEBUG)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Weffc++")
ENDIF()

# This is for gnu c++ compiler OPENMP support.
# We have to compile with vec_features (to avoid shared object)
if(OPENMP GREATER 0)
  message("Compiling with openmp support")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
  ADD_DEFINITIONS(-D_VEC_FEATURES)
endif()

if (PROFILING GREATER 0)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
endif()

# We need to add other subdirectory
if (NOT _PUBLIC)
	ADD_SUBDIRECTORY(${PROJECT_SOURCE_DIR}/../../3rdParty/asa  asa)
	ADD_SUBDIRECTORY(${PROJECT_SOURCE_DIR}/../../3rdParty/owlqn  owlqn)
endif()
ADD_SUBDIRECTORY(${PROJECT_SOURCE_DIR}/../../3rdParty/cgDescent  cgdescent)
ADD_SUBDIRECTORY(${PROJECT_SOURCE_DIR}/../../3rdParty/liblbfgs  lbfgs)
ADD_SUBDIRECTORY(${PROJECT_SOURCE_DIR}/../../3rdParty/uncOptim  uncOptim)
# The file we need to include
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/include)
if (NOT _PUBLIC)
INCLUDE_DIRECTORIES(${ASA_SOURCE_DIR}/include)
INCLUDE_DIRECTORIES(${owlqn_SOURCE_DIR}/include)
endif()
INCLUDE_DIRECTORIES(${cgDescent_SOURCE_DIR}/include)
INCLUDE_DIRECTORIES(${liblbfgs_SOURCE_DIR}/include)
INCLUDE_DIRECTORIES(${uncOptim_SOURCE_DIR}/include)

# This is the list of source files
SET(LIB_SRC src/BackwardWindowRawFeatures.cpp
src/DataSequenceRealtime.cpp
src/dataset.cpp
src/DistEdgeFeaturesMV.cpp
src/Dummys.cpp
src/EdgeFeatures.cpp
src/EdgeObservationFeatures.cpp
src/evaluator.cpp
src/evaluatorCRF.cpp
src/evaluatorHCRF.cpp
src/evaluatorLDCRF.cpp
src/evaluatorLVPERCEPTRON.cpp
src/evaluatorMVHCRF.cpp
src/evaluatorMVLDCRF.cpp
src/EvaluatorSharedLDCRF.cpp
src/featuregenerator.cpp
src/FeaturesOne.cpp
src/GateNodeFeatures.cpp
src/GaussianWindowRawFeatures.cpp
src/gradient.cpp
src/GradientCNF.cpp
src/GradientCRF.cpp
src/GradientFD.cpp
src/GradientHCNF.cpp
src/GradientHCRF.cpp
src/GradientHMMPerceptron.cpp
src/GradientLDCNF.cpp
src/GradientLDCRF.cpp
src/GradientMVHCRF.cpp
src/GradientMVLDCRF.cpp
src/GradientPerceptron.cpp
src/GradientSharedLDCRF.cpp
#src/GradientSHDCRF.cpp
src/InferenceEngine.cpp
src/InferenceEngineBP.cpp
src/InferenceEngineBrute.cpp
src/InferenceEngineDC.cpp
src/InferenceEngineFB.cpp
src/InferenceEngineFF.cpp
#src/InferenceEngineGBP.cpp
src/InferenceEngineJT.cpp
src/InferenceEngineLoopyBP.cpp
src/InferenceEnginePerceptron.cpp
src/LabelEdgeFeatures.cpp
src/matrix.cpp
src/matrixSSE2.cpp
src/model.cpp
src/MultiviewFeatures.cpp
src/MyFeatures.cpp
src/MyLineSearchWolfe.cpp
src/MyQPSolver.cpp
src/MyToolbox.cpp
src/optimizer.cpp
src/OptimizerASA.cpp
src/OptimizerCG.cpp
src/OptimizerLBFGS.cpp
src/OptimizerNRBM.cpp
src/OptimizerOWL.cpp
src/OptimizerPerceptron.cpp
src/OptimizerUncOptim.cpp
src/RawFeatures.cpp
src/RawFeaturesSquare.cpp
src/SharedFeatures.cpp
src/StartFeatures.cpp
src/toolbox.cpp
src/ToolboxCRF.cpp
src/ToolboxCRFRealtime.cpp
src/ToolboxGHCRF.cpp
src/ToolboxHCRF.cpp
src/ToolboxHMMPerceptron.cpp
src/ToolboxLDCRF.cpp
src/ToolboxLVPERCEPTRON.cpp
src/ToolboxMVHCRF.cpp
src/ToolboxMVLDCRF.cpp
src/ToolboxSharedLDCRF.cpp
src/WindowRawFeatures.cpp
) 

#Simple way to add all cpp files, but some cpps added to the folder simply
#don't compile, so I'll leave it the previous way.
#FILE(GLOB src_files src/*.cpp)
#SOURCE_GROUP("Source files" FILES ${src_files})

add_library(hCRF ${LIB_SRC})# ${src_files})

SET(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR})
INSTALL_TARGETS(/lib hCRF)
