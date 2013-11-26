# Intermediary files
BUILD_DIR := obj

# Target program
TARGET := prog.x

# Any includes
SRC_INCDIRS := inc/ ~/work/boost_1_51_0/
INCDIRS := inc/ ~/work/boost/1_51_0


# Move the binary to bin/
define MOVE
	mv ${TARGET} bin/
endef	
TGT_POSTMAKE := ${MOVE}

# Submakefiles
SUBMAKEFILES := src/root.mk
