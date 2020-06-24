# testing

runminimal_1:
	-@../testit.sh minimal.py "FIXME" 1 1

test_minimal: runminimal_1

test: test_minimal

# etc

.PHONY: clean runminimal_1 test_minimal test

clean:
	@rm -f *.pyc *.geo *.msh *.pvd *.vtu maketmp tmp difftmp

