surface/
========

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

### sphere.py

This silly code simply uses Firedrake to compute a parameterized sphere suitable for viewing in Paraview.

### blow.py

The above `minimal.py` code solves a scalar, nonlinear, elliptic PDE for a surface z = u(x,y).  This one solves the same problem but in parametric form, that is, for a surface in the form x = x(s,t), y = y(s,t), z = z(s,t).  The first-variation of the area functional is made coercive by adding Laplacians to x,y.  (Which only works because we know the surface is a z = u(x,y).)

### surface.py.rst

UNDER CONSTRUCTION

View the `pylit`/`reStructuredText` document this way:

        $ restview surface.py.rst

Extract a run-able code and run this way:

        (firedrake) $ python2 ../pylit/pylit.py surface.py.rst
        extract written to surface.py
        (firedrake) $ python3 surface.py

It writes files `surfaceX.pvd` for `X`=1,2,3.  View the result with Paraview
and use Warp By Vector.

### references

* E. Bueler, PETSc for Partial Differential Equations: Numerical solutions in C and Python, SIAM Press, 2020?

### testing

    $ make test

### cleaning up

    $ make clean

