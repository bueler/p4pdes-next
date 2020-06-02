riemann/
========

### compile and run

The code is `riemann.c`.  Do this to build and run:

        $ make riemann
        $ ./riemann -help intro     # basic info
        $ mpiexec -n 2 ./riemann    # parallel solution

As an example of what it can do see this shallow water equations solution of
a dam break problem:

        $ ./riemann -da_grid_x 2000 -limiter mc -problem swater -initial dam \
            -ts_monitor_solution draw -draw_size 1000,200

### testing

    $ make test

### cleaning up

    $ make clean

