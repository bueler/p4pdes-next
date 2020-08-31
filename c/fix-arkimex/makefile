include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
CFLAGS += -pedantic -std=c99

ex54: ex54.o
	-${CLINKER} -o ex54 ex54.o  ${PETSC_LIB}
	${RM} ex54.o

# testing

ex54_1:
	-@../../testit.sh ex54 "" 1 1 "basic BDF2"

ex54_2:
	-@../../testit.sh ex54 "-ts_type arkimex -identity_in_F" 1 2 "ARKIMEX needs dF/d(dudt) = I"

ex54_3:
	-@../../testit.sh ex54 "-ts_rtol 1.0e-10 -ts_atol 1.0e-10 -ts_max_time 0.001" 1 3 "ask for accuracy"

test: ex54_1 ex54_2 ex54_3

# etc

.PHONY: clean ex54_1 ex54_2 ex54_3 test

clean::
	@rm -f *~ ex54 difftmp maketmp tmp

