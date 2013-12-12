SOURCES := main.cu

CXXFLAGS := -O0 -g -G -arch sm_20  
#CXXFLAGS := -O0 -g -G 

#SUBMAKEFILES := linear_algebra/containers/containers.mk

SRC_INCDIRS := ../inc/

LDFLAGS := -O0 -g -G -arch sm_20 
