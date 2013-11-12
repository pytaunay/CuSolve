# Intermediary files
BUILD_DIR := obj

# Target program
TARGET := prog.x

# Any includes
SRC_INCDIRS := inc/
INCDIRS := inc/


# Move the binary to bin/
define MOVE
	mv ${TARGET} bin/
endef	
TGT_POSTMAKE := ${MOVE}

# Submakefiles
SUBMAKEFILES := src/root.mk
