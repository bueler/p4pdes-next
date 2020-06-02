include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
CFLAGS += -pedantic -std=c99

riemann: riemann.o
	-${CLINKER} -o riemann riemann.o  ${PETSC_LIB}
	${RM} riemann.o

# testing

riemann_1:
	-@../testit.sh riemann "-da_grid_x 5 -ts_monitor" 1 1

riemann_2:
	-@../testit.sh riemann "-problem acoustic -da_grid_x 100 -ts_max_time 3 -ts_type rk -ts_rk_type 5bs" 2 2

riemann_3:
	-@../testit.sh riemann "-problem swater -da_grid_x 10 -ts_monitor" 1 3

riemann_4:
	-@../testit.sh riemann "-problem swater -initial dam -da_grid_x 5 -ts_max_time 2 -ts_monitor" 1 4

riemann_5:
	-@../testit.sh riemann "-problem swater -da_grid_x 7 -ts_dt 0.1 -ts_max_time 0.2 -ts_monitor" 1 5

riemann_6:
	-@../testit.sh riemann "-problem swater -limiter minmod -initial dam -da_grid_x 8" 4 6

riemann_7:
	-@../testit.sh riemann "-problem acoustic -limiter mc -initial stump -da_grid_x 10" 1 7

riemann_8:
	-@../testit.sh riemann "-problem advection -limiter fromm -da_grid_x 9" 1 8

riemann_9:
	-@../testit.sh riemann "-problem traffic -initial greenlight -da_grid_x 5 -ts_monitor" 1 9

riemann_10:
	-@../testit.sh riemann "-problem traffic -limiter mc -da_grid_x 20 -ts_max_time 5 -ts_dt 1 -ts_monitor" 1 10

test: riemann_1 riemann_2 riemann_3 riemann_4 riemann_5 riemann_6 riemann_7 riemann_8 riemann_9 riemann_10

# etc

.PHONY: clean riemann_1 riemann_2 riemann_3 riemann_4 riemann_5 riemann_6 riemann_7 riemann_8 riemann_9 riemann_10 test

clean::
	@rm -f *~ riemann difftmp maketmp tmp

