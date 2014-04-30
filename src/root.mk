SOURCES := main.cu roberts.cu hires.cu orego.cu e5.cu
#SOURCES := main.cu ali.cu

CXXFLAGS := -O3 -arch sm_35 -D__VERBOSE 
#CXXFLAGS := -O0 -g -G -arch sm_35 -D__VERBOSE 

#SUBMAKEFILES := linear_algebra/containers/containers.mk

SRC_INCDIRS := ../inc/

#LDFLAGS := -g -G -arch sm_35 -D__VERBOSE -L/gpfs/apps/cuda-rhel6/cuda/5.5/lib64 -lcublas
LDFLAGS := -O3 -arch sm_35 -D__VERBOSE -L/gpfs/apps/cuda-rhel6/cuda/5.5/lib64 -lcublas
