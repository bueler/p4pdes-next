riemann/
========

### compile and run

The code is `riemann.c`.  Do this to build and run:

        $ make riemann
        $ ./riemann -help intro     # basic info
        $ mpiexec -n 2 ./riemann    # parallel solution

As an example of what it can do see this high-resolution shallow water solution
of a dam break problem using SSP time-stepping:

        $ ./riemann -da_grid_x 3000 -limiter mc -problem swater -initial dam \
            -ts_type ssp -ts_monitor_solution draw -draw_size 1000,200

### references

* E. Bueler, PETSc for Partial Differential Equations: Numerical solutions in C and Python, SIAM Press, 2020?

* S. Gottlieb, C. W. Shu, & E. Tadmor, Strong stability-preserving high-order time discretization methods. SIAM Review, 43(1), 89-112, 2001

* R. J. LeVeque, Finite Volume Methods for Hyperbolic Problems, Cambridge University Press, 2002

### testing

    $ make test

### cleaning up

    $ make clean

