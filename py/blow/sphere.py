#!/usr/bin/env python3

# creates a parameterized surface and saves it to sphere.pvd for viewing
# using Paraview's Warp By Vector

from firedrake import *
from firedrake.petsc import PETSc

# triangulation on (s,t) in [0,1]x[0,1]
ms, mt = 20, 20
mesh = UnitSquareMesh(ms-1, mt-1)

# surface:  X(s,t) = (x(s,t), y(s,t), z(s,t))
V = VectorFunctionSpace(mesh, 'Lagrange', degree=1, dim=3)
X = Function(V)

# to show the sphere we must remove the unit square
s,t = SpatialCoordinate(mesh)
Xtrue = Function(V).interpolate(as_vector([sin(pi*t)*cos(2*pi*s) - s, \
                                           sin(pi*t)*sin(2*pi*s) - t, \
                                           cos(pi*t)]))

# silly: solve weak form of
#     X(s,t) = Xtrue(s,t)
# reference: "A first variational form" at
# https://www.firedrakeproject.org/variational-problems.html
Phi = TestFunction(V)
F = (inner(X,Phi) - inner(Xtrue,Phi)) * dx

# solve nonlinear system:  F(u) = 0
solve(F == 0, X, options_prefix = 's',
      solver_parameters = {'snes_type': 'ksponly',
                           'ksp_type': 'preonly',
                           'pc_type': 'lu'})

# optional dump
if False:
    with X.dat.vec_ro as vX:
        vX.view()

# silly: print numerical error in L_infty norm
Xdiff = Function(V).interpolate(X - Xtrue)
with Xdiff.dat.vec_ro as vXdiff:
    error_Linf = abs(vXdiff).max()[1]
PETSc.Sys.Print('done on %d x %d grid:  error |u-uexact|_inf = %.3e' \
      % (ms,mt,error_Linf))

# to view surface in Paraview:
#     * use Warp By Vector with Scale Factor 1
#     * Change Interaction Mode to 3D so you can rotate
#     * try Reset or Zoom To Data
name = 'sphere.pvd'
PETSc.Sys.Print('saving to %s ...' % name)
X.rename('X(s,t)')
File(name).write(X)

