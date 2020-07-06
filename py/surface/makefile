# testing

runminimal_1:
	-@../../testit.sh minimal.py "-refine 2 -s_snes_converged_reason -s_ksp_converged_reason -s_pc_type mg" 1 1

runminimal_2:
	-@../../testit.sh minimal.py "-refine 1 -sequence 1 -s_pc_type gamg -s_snes_converged_reason -s_ksp_converged_reason" 2 2

runminimal_3:
	-@../../testit.sh minimal.py "-sequence 2 -k 2 -s_snes_converged_reason -s_pc_type mg" 1 3

runblow_1:
	-@../../testit.sh blow.py "-s_pc_type mg -s_snes_converged_reason -s_ksp_converged_reason -refine 1 -sequence 1" 1 1

surface.py:
	python2 ../pylit/pylit.py surface.py.rst

# FIXME this may not work because surface.py is not executable?
runsurface_1: surface.py
	-@../../testit.sh surface.py "" 1 1

test_minimal: runminimal_1 runminimal_2 runminimal_3

test_blow: runblow_1

test_surface: runsurface_1

# FIXME  add test_surface
test: test_minimal test_blow

# etc

.PHONY: clean runminimal_1 runminimal_2 runminimal_3 test_minimal runblow_1 test_blow runsurface_1 test_surface test

clean:
	@rm -f *.pyc *.geo *.msh *.pvd *.vtu maketmp tmp difftmp

