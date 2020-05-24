include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
CFLAGS += -pedantic -std=c99

riemann: riemann.o
	-${CLINKER} -o riemann riemann.o  ${PETSC_LIB}
	${RM} riemann.o

# etc

.PHONY: distclean

distclean:
	@rm -f *~ riemann

