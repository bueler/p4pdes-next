#!/usr/bin/env python3

#FIXME: document better, here and in a .tex for the derivation of the weak form

#FIXME: add a minimal surface that is not so boring, with x(s,t), y(s,t) less
#       trivial

from argparse import ArgumentParser, RawTextHelpFormatter
from firedrake import *
from firedrake.petsc import PETSc

parser = ArgumentParser(description="""
Solves a parameterized-surface version of the minimal surface equation.
Compare minimal.py.""",
    formatter_class=RawTextHelpFormatter,add_help=False)
parser.add_argument('-blowhelp', action='store_true', default=False,
                    help='help for blow.py options')
parser.add_argument('-k', type=int, default=1, metavar='K',
                    help='polynomial degree for Q_k elements')
parser.add_argument('-ms', type=int, default=3, metavar='MS',
                    help='number of (coarse) grid points in s-direction')
parser.add_argument('-mt', type=int, default=3, metavar='MT',
                    help='number of (coarse) grid points in t-direction')
parser.add_argument('-o', metavar='NAME', type=str, default='',
                    help='output file name ending with .pvd')
parser.add_argument('-printcoords', action='store_true', default=False,
                    help='print coordinates of mesh nodes')
parser.add_argument('-refine', type=int, default=0, metavar='N',
                    help='number of refinement levels (determines base grid for -sequence)')
parser.add_argument('-sequence', type=int, default=0, metavar='N',
                    help='number of grid-sequencing levels')
parser.add_argument('-zeroinitial', action='store_true', default=False,
                    help='initialize with X(s,t)=0 (including x,y)')
args, unknown = parser.parse_known_args()
assert (args.k >= 1)
assert (args.ms >= 2)
assert (args.mt >= 2)
assert (args.refine >= 0)
assert (args.sequence >= 0)
if args.blowhelp:
    parser.print_help()

# Create mesh in parameter space (s,t)
ms, mt = args.ms, args.mt
mesh = UnitSquareMesh(ms-1, mt-1, quadrilateral=True)

# Enable GMG by refinement hierarchy, and grid-sequencing by further refinement
hierarchy = MeshHierarchy(mesh, args.refine + args.sequence)
if args.refine > 0:
    mesh = hierarchy[args.refine]
    ms, mt = (ms-1) * 2**args.refine + 1, (mt-1) * 2**args.refine + 1

# Two views of the mesh
mesh._topology_dm.viewFromOptions('-dm_view')
#V.dm.viewFromOptions('-dm_view')  # dumps a Vec, unfortunately
if args.printcoords:
    print(mesh.coordinates.dat.data)

# Function space for a parameterized surface, and initial guess
V = VectorFunctionSpace(mesh, 'Lagrange', degree=args.k, dim=3)
if args.zeroinitial:
    X = Function(V)
else:
    s,t = SpatialCoordinate(mesh)
    X = Function(V).interpolate(as_vector([s,t,1]))   # z=1 much better than z=0 ... why?

# Grid-sequencing loop;  replaces -snes_grid_sequence in PETSc codes
for j in range(args.sequence+1):    # always runs once
    # Define weak form
    Phi = TestFunction(V)
    prod = cross(X.dx(0),X.dx(1))
    N = prod / sqrt(inner(prod,prod))  # normal vector to surface
    #FIXME: consider regularization here:
    #N = prod / sqrt(1.0e-6 + inner(prod,prod))
    F = inner(grad(X[0]),grad(Phi[0])) * dx \
        + inner(grad(X[1]),grad(Phi[1])) * dx \
        + inner(N,cross(X.dx(0),Phi.dx(1)) - cross(X.dx(1),Phi.dx(0))) * dx

    # Define Dirichlet boundary conditions, also the exact solution
    c = 1.1  # see example in Chapter 7 of Bueler, PETSc for PDEs
    s,t = SpatialCoordinate(mesh)
    g_bdry = c * cosh(s/c) * sin(acos( (t/c) / cosh(s/c) ))
    bdry_ids = (1, 2, 3, 4)   # all four sides of boundary are Dirichlet
    bcx = DirichletBC(V.sub(0), s, bdry_ids)
    bcy = DirichletBC(V.sub(1), t, bdry_ids)
    bcz = DirichletBC(V.sub(2), g_bdry, bdry_ids)

    # Solve nonlinear system:  F(u) = 0
    solve(F == 0, X, bcs = [bcx,bcy,bcz], options_prefix = 's',
          solver_parameters = {'snes_type': 'newtonls',
                               'ksp_type': 'cg'})

    # Print numerical error in L_infty norm
    udiff = Function(V).interpolate(X - as_vector([s,t,g_bdry]))
    with udiff.dat.vec_ro as vudiff:
        error_Linf = abs(vudiff).max()[1]
    spaces = (args.sequence - j) * '  '
    PETSc.Sys.Print('%sdone on %d x %d grid of Q_%d:  error |u-uexact|_inf = %.3e' \
          % (spaces,ms,mt,args.k,error_Linf))

    # Generate initial iterate at next level by interpolation from solution
    if j < args.sequence:
        Xcoarse = X.copy()
        mesh = hierarchy[args.refine+j+1]
        ms, mt = (ms-1) * 2 + 1, (mt-1) * 2 + 1
        mesh._topology_dm.viewFromOptions('-dm_view')
        V = VectorFunctionSpace(mesh, 'Lagrange', degree=args.k, dim=3)
        X = Function(V)
        prolong(Xcoarse,X)

# Optionally save to a .pvd file viewable with Paraview
if len(args.o) > 0:
    PETSc.Sys.Print('saving solution to %s ...' % args.o)
    X.rename('X(s,t)')
    File(args.o).write(X)

