DEPTH = /opt/AMDAPP

include $(DEPTH)/make/openclsdkdefs.mk 

####
#
#  Targets
#
####

OPENCL			= 1
SAMPLE_EXE		= 1
EXE_TARGET 		= descomposicionLU
EXE_TARGET_INSTALL   	= descomposicionLU

####
#
#  C/CPP files
#
####

FILES 	= descomposicionLU
CLFILES = descomposicionLU_kernels.cl 

LLIBS  	+= SDKUtil

include $(DEPTH)/make/openclsdkrules.mk 

copy:
	cp /opt/AMDAPP/samples/opencl/myprojects/app/luProject/descomposicion-lu/FactorizacionLUKernels.cl /opt/AMDAPP/samples/opencl/myprojects/app/luProject/descomposicion-lu/build/debug/x86_64
