include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
CFLAGS += -pedantic -std=c99

riemann: riemann.o
	-${CLINKER} -o riemann riemann.o  ${PETSC_LIB}
	${RM} riemann.o

# use this target to create symbolic links to PETSc binary files scripts
petscPyScripts:
	ln -sf ${PETSC_DIR}/lib/petsc/bin/PetscBinaryIO.py
	ln -sf ${PETSC_DIR}/lib/petsc/bin/petsc_conf.py

# testing

riemann_1:
	-@../../testit.sh riemann "-da_grid_x 5 -ts_monitor" 1 1 "basic acoustics"

riemann_2:
	-@../../testit.sh riemann "-problem acoustic -da_grid_x 100 -ts_max_time 3 -ts_type rk -ts_rk_type 5bs" 2 2 "parallel, acoustics with better RK"

riemann_3:
	-@../../testit.sh riemann "-problem swater -da_grid_x 10 -ts_monitor" 1 3 "basic shallow water"

riemann_4:
	-@../../testit.sh riemann "-problem swater -initial dam -da_grid_x 5 -ts_max_time 2 -ts_monitor" 1 4 "shallow water dam problem"

riemann_5:
	-@../../testit.sh riemann "-problem swater -b0 -1 -bx -0.1 -initial hump -limiter none -ts_type ssp -da_grid_x 20" 1 5 "shallow water with bottom slope"

riemann_6:
	-@../../testit.sh riemann "-problem swater -limiter minmod -initial dam -da_grid_x 8" 4 6 "parallel, shallow water with minmod limiter"

riemann_7:
	-@../../testit.sh riemann "-problem acoustic -limiter mc -initial stump -da_grid_x 10" 1 7 "acoustic with mc limiter"

riemann_8:
	-@../../testit.sh riemann "-problem advection -limiter fromm -da_grid_x 9" 1 8 "advection with fromm limiter"

riemann_9:
	-@../../testit.sh riemann "-problem traffic -initial greenlight -da_grid_x 5 -ts_monitor" 1 9 "traffic greenlight problem"

riemann_10:
	-@../../testit.sh riemann "-problem traffic -limiter mc -da_grid_x 20 -ts_max_time 5 -ts_dt 1 -ts_monitor" 1 10 "traffic with mc limiter"


test: riemann_1 riemann_2 riemann_3 riemann_4 riemann_5 riemann_6 riemann_7 riemann_8 riemann_9 riemann_10

# etc

.PHONY: clean riemann_1 riemann_2 riemann_3 riemann_4 riemann_5 riemann_6 riemann_7 riemann_8 riemann_9 riemann_10 test petscPyScripts

clean::
	@rm -f *~ riemann difftmp maketmp tmp
	@rm -f PetscBinaryIO.py petsc_conf.py *.png *.dat *.dat.info
	@rm -rf __pycache__/

