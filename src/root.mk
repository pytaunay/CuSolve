SOURCES := main.cu

CXXFLAGS := -O3 -arch sm_30 -D__VERBOSE 
#CXXFLAGS := -O3 -g -G -arch sm_30 -D__VERBOSE 
#CXXFLAGS := -O0 -g -G 

#SUBMAKEFILES := linear_algebra/containers/containers.mk

SRC_INCDIRS := ../inc/

LDFLAGS := -O3 -arch sm_30 -D__VERBOSE
#LDFLAGS := -O3 -g -G -arch sm_30 -D__VERBOSE
