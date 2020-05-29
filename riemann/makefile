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
	-@../testit.sh riemann "-problem acoustic -da_grid_x 100 -ts_max_time 3 -ts_type rk -ts_rk_type 5bs" 2 2

runriemann_3:
	-@../testit.sh riemann "-problem swater -da_grid_x 10 -ts_monitor" 1 3

runriemann_4:
	-@../testit.sh riemann "-problem swater -initial dam -da_grid_x 5 -ts_max_time 2 -ts_monitor" 1 4

runriemann_5:
	-@../testit.sh riemann "-problem swater -da_grid_x 7 -ts_dt 0.1 -ts_max_time 0.2 -ts_monitor" 1 5

runriemann_6:
	-@../testit.sh riemann "-problem swater -limiter minmod -initial dam -da_grid_x 8" 4 6

test: runriemann_1 runriemann_2 runriemann_3 runriemann_4 runriemann_5 runriemann_6

# etc

.PHONY: clean runriemann_1 runriemann_2 runriemann_3 runriemann_4 runriemann_5 runriemann_6 test

clean::
	@rm -f *~ riemann difftmp maketmp tmp

