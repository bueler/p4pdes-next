riemann/
========

### compile and run

The PETSc code here is `riemann.c`.  The problems it solves are in several `.h` files; see `cases.h` for more.

Do this to build and run the default example from `acoustic.h`:

        $ make riemann
        $ ./riemann -help intro     # basic info
        $ ./riemann -da_grid_x 2000 -ts_monitor_solution draw \  # runtime movie
            -draw_size 1000,200 -draw_pause 0.01
        $ mpiexec -n 2 ./riemann    # parallel solution

As an example of what it can do see this high-resolution shallow water solution of a dam break problem using SSP time-stepping:

        $ ./riemann -problem swater -initial dam -limiter mc -ts_type ssp \
            -da_grid_x 3000 -ts_monitor_solution draw -draw_size 1000,200

### generating movies

The scripts `plotTS.py` and `plotsw.py` can be used to generate `.png` image files for the frames of a movie.  This requires saving PETSc binary files for the t-axis (`-ts_monitor binary:t.dat`) and for the solution (`-ts_monitor_solution binary:q.dat`).  Furthermore it requires a link to the PETSc script `PetscBinaryIO.py`.

For example, the following generates a movie from the `traffic.h` case:

        $ ./riemann -problem traffic -da_grid_x 1000 -limiter mc \
            -ts_monitor_solution binary:q.dat -ts_monitor binary:t.dat
        $ mkdir traffic
        $ make petscPyScripts  # link to PetscBinaryIO.py
        $ ./plotTS.py -mx 1000 -ylabel "density" -ax -30.0 -bx 30.0 \
            -cellcentered -oroot traffic/bar t.dat q.dat

This generates files `traffic/barXXX.png` which can be viewed with an image viewer.  To actually generate a `.m4v` or similar movie, consider using [`ffmpeg`](https://www.ffmpeg.org/).

The similar image-generating script `plotsw.py` is able to correctly display the water surface.  (Note eta = h + B; see `swater.h`.)  For example:

        $ FIXME

### references

* E. Bueler, PETSc for Partial Differential Equations: Numerical solutions in C and Python, SIAM Press, 2020?

* S. Gottlieb, D. I. Ketcheson, & C. W. Shu, High order strong stability preserving time discretizations. Journal of Scientific Computing, 38(3), 251-289, 2009

* R. J. LeVeque, Finite Volume Methods for Hyperbolic Problems, Cambridge University Press, 2002

### testing

    $ make test

### cleaning up

    $ make clean

