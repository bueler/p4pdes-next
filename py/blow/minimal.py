#!/usr/bin/env python3

from argparse import ArgumentParser, RawTextHelpFormatter
from firedrake import *
from firedrake.petsc import PETSc

parser = ArgumentParser(description="""
Use Firedrake's nonlinear solver for the minimial surface problem
  - div ( (1 + |grad u|^2)^q grad u ) = 0
on the unit square S=(0,1)^2 subject to Dirichlet boundary conditions
u = g(x,y).  Power q defaults to -1/2, but it can be set using -q.
Catenoid boundary conditions are implemented; this is an exact solution.
(Compare c/ch7/minimal.c at https://github.com/bueler/p4pdes.)
The discretization is by Q_k finite elements; the default is k=1 but it
can be set by using -k.  This code is multigrid (GMG) capable (-s_pc_type mg)
and it implements grid-sequencing by replacing the functionality of
-snes_grid_sequence in PETSc codes.  The prefix for PETSc solver options is
's_'.  Use -help for PETSc options and -minhelp for options to this program.""",
    formatter_class=RawTextHelpFormatter,add_help=False)
parser.add_argument('-minhelp', action='store_true', default=False,
                    help='help for minimal.py options')
parser.add_argument('-k', type=int, default=1, metavar='K',
                    help='polynomial degree for Q_k elements')
parser.add_argument('-mx', type=int, default=3, metavar='MX',
                    help='number of (coarse) grid points in x-direction')
parser.add_argument('-my', type=int, default=3, metavar='MY',
                    help='number of (coarse) grid points in y-direction')
parser.add_argument('-o', metavar='NAME', type=str, default='',
                    help='output file name ending with .pvd')
parser.add_argument('-q', type=float, default=-0.5, metavar='Q',
                    help='exponent in coefficient')
parser.add_argument('-refine', type=int, default=0, metavar='N',
                    help='number of refinement levels (determines base grid for -sequence)')
parser.add_argument('-sequence', type=int, default=0, metavar='N',
                    help='number of grid-sequencing levels')
args, unknown = parser.parse_known_args()
assert (args.k >= 1)
assert (args.mx >= 2)
assert (args.my >= 2)
assert (args.refine >= 0)
assert (args.sequence >= 0)
if args.minhelp:
    parser.print_help()

# Create mesh
mx, my = args.mx, args.my
mesh = UnitSquareMesh(mx-1, my-1, quadrilateral=True)

# Enable GMG by refinement hierarchy, and grid-sequencing by further refinement
if args.refine + args.sequence > 0:
    hierarchy = MeshHierarchy(mesh, args.refine + args.sequence)
if args.refine > 0:
    mesh = hierarchy[args.refine]
    mx, my = (mx-1) * 2**args.refine + 1, (my-1) * 2**args.refine + 1

# to view mesh:  mesh._plex.viewFromOptions('-dm_view')
#                print(mesh.coordinates.dat.data)

# Grid-sequencing loop
W = FunctionSpace(mesh, 'Lagrange', degree=args.k)
u = Function(W)  # initialized to zero here
for j in range(args.sequence):
    #PETSc.Viewer.STDOUT(comm=PETSc.Object(PETSc.SNES).getComm()).pushASCIITab() seg faults
    PETSc.Viewer.STDOUT().pushASCIITab() # does not affect tabs from e.g. -snes_converged_reason
for j in range(args.sequence+1):    # always runs once
    # Define weak form
    v = TestFunction(W)
    F = ((1.0 + dot(grad(u),grad(u)))**args.q * dot(grad(u), grad(v))) * dx

    # Define Dirichlet boundary conditions, also the exact solution
    c = 1.1  # see example in Chapter 7 of Bueler, PETSc for PDEs
    x,y = SpatialCoordinate(mesh)
    g_bdry = Function(W).interpolate(c * cosh(x/c) * sin(acos( (y/c) / cosh(x/c) )))
    bdry_ids = (1, 2, 3, 4)   # all four sides of boundary are Dirichlet
    bc = DirichletBC(W, g_bdry, bdry_ids)

    PETSc.Viewer.STDOUT().printfASCII('')  # FIXME hack to affect tabs

    # Solve nonlinear system:  F(u) = 0
    solve(F == 0, u, bcs = [bc], options_prefix = 's',
          solver_parameters = {'snes_type': 'newtonls',
                               'ksp_type': 'cg'})

    # Generate initial iterate at next level by interpolation from solution
    if j < args.sequence:
        ucoarse = u.copy()
        mesh = hierarchy[args.refine+j+1]
        mx, my = (mx-1) * 2 + 1, (my-1) * 2 + 1
        W = FunctionSpace(mesh, 'Lagrange', degree=args.k)
        u = Function(W)
        prolong(ucoarse,u)
        PETSc.Viewer.STDOUT().popASCIITab()

# Print numerical error in L_infty norm
udiff = Function(W).interpolate(u - g_bdry)
with udiff.dat.vec_ro as vudiff:
    error_Linf = abs(vudiff).max()[1]
PETSc.Sys.Print('done on %d x %d grid of Q_%d:  error |u-uexact|_inf = %.3e' \
      % (mx,my,args.k,error_Linf))

# Optionally save to a .pvd file viewable with Paraview
if len(args.o) > 0:
    PETSc.Sys.Print('saving solution to %s ...' % args.o)
    u.rename('u')
    File(args.o).write(u)

