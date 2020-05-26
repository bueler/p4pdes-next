include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
CFLAGS += -pedantic -std=c99

riemann: riemann.o
	-${CLINKER} -o riemann riemann.o  ${PETSC_LIB}
	${RM} riemann.o

# testing

runriemann_1:
	-@../testit.sh riemann "-da_grid_x 5 -ts_monitor" 1 1

runriemann_2:
	-@../testit.sh riemann "-da_grid_x 100 -ts_max_time 3 -ts_type rk -ts_rk_type 5bs" 2 2

test: runriemann_1 runriemann_2

# etc

.PHONY: distclean runriemann_1 runriemann_2 test

distclean:
	@rm -f *~ riemann

