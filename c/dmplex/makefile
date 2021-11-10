include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
CFLAGS += -pedantic -std=c99

meshtest: meshtest.o
	-${CLINKER} -o meshtest meshtest.o ${PETSC_LIB}
	${RM} meshtest.o

# etc

.PHONY: distclean

distclean:
	@rm -f *~ meshtest

