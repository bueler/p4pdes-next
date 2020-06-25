# testing

runminimal_1:
	-@../testit.sh minimal.py "-refine 2 -s_snes_converged_reason -s_ksp_converged_reason -s_pc_type mg" 1 1

test_minimal: runminimal_1

test: test_minimal

# etc

.PHONY: clean runminimal_1 test_minimal test

clean:
	@rm -f *.pyc *.geo *.msh *.pvd *.vtu maketmp tmp difftmp

