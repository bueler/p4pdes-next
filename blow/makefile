# testing

runminimal_1:
	-@../testit.sh minimal.py "-refine 2 -s_snes_converged_reason -s_ksp_converged_reason -s_pc_type mg" 1 1

runminimal_2:
	-@../testit.sh minimal.py "-refine 1 -sequence 1 -s_pc_type gamg -s_snes_converged_reason -s_ksp_converged_reason" 2 2

runminimal_3:
	-@../testit.sh minimal.py "-sequence 2 -k 2 -s_snes_converged_reason -s_pc_type mg" 1 3

test_minimal: runminimal_1 runminimal_2 runminimal_3

test: test_minimal

# etc

.PHONY: clean runminimal_1 runminimal_2 runminimal_3 test_minimal test

clean:
	@rm -f *.pyc *.geo *.msh *.pvd *.vtu maketmp tmp difftmp

