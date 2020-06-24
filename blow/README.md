blow/
=====

### minimial.py

This code solves the same problem as `minimal.c` in Chapter 7 of [PETSc for PDEs](https://github.com/bueler/p4pdes).  That is, it solves a scalar, nonlinear, elliptic PDE for a surface z=u(x,y).  To run it do

        $ source firedrake/bin/activate    # on Bueler's machine: drakeme
        $ ./minimal.py

Get help with these forms:

        $ ./minimal.py -minhelp   # this program
        $ ./minimal.py -help      # PETSc options

### blow.py

FIXME The plan for this code is that it solves the minimal surface equation but in parametric form, that is, for a surface in the form x = x(s,t), y = y(s,t), z = z(s,t). 

### references

* E. Bueler, PETSc for Partial Differential Equations: Numerical solutions in C and Python, SIAM Press, 2020?

### testing

    $ make test  # FIXME

### cleaning up

    $ make clean

