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
	-@../../testit.sh ex54 "-ts_type arkimex" 1 2 "wrong values with ARKIMEX"

test: ex54_1 ex54_2

# etc

.PHONY: clean ex54_1 ex54_2 test

clean::
	@rm -f *~ ex54 difftmp maketmp tmp

