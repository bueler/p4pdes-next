blow/
=====

### minimal.py

This code solves the same "catenoid" problem as `minimal.c` in Chapter 7 of [PETSc for PDEs](https://github.com/bueler/p4pdes), but here we use Q_k finite elements.  To run it do

        $ source firedrake/bin/activate    # on Bueler's machine: drakeme
        (firedrake) $ ./minimal.py

Get help with these forms:

        (firedrake) $ ./minimal.py -minhelp   # this program
        (firedrake) $ ./minimal.py -help      # PETSc options

To see a high-accuracy example using a full nonlinear multigrid cycle, i.e. grid-sequenced Newton-multigrid, using Q_2 elements, do something like

        (firedrake) $ ./minimal.py -k 2 -sequence 8 -s_pc_type mg

The finest grid here is 513x513 and 10 digit accuracy is achieved at all nodes.

### blow.py

FIXME The above code `minimal.py` solves a scalar, nonlinear, elliptic PDE for a surface z = u(x,y).  The plan for this code is that it solves the minimal surface equation but in parametric form, that is, for a surface in the form x = x(s,t), y = y(s,t), z = z(s,t).

### references

* E. Bueler, PETSc for Partial Differential Equations: Numerical solutions in C and Python, SIAM Press, 2020?

### testing

    $ make test

### cleaning up

    $ make clean

