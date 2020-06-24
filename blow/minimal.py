#!/usr/bin/env python3

from argparse import ArgumentParser, RawTextHelpFormatter
from firedrake import *
from firedrake.petsc import PETSc

parser = ArgumentParser(description="""
Use Firedrake's nonlinear solver for the minimial surface problem
  - div ( (1 + |grad u|^2)^q grad u ) = 0
on the unit square S=(0,1)^2 subject to Dirichlet boundary
conditions u = g(x,y).  Power q defaults to -1/2 but can be set (by -q).
Catenoid boundary conditions are implemented; this is an exact solution.
capable.  Compare c/ch7/minimal.c at https://github.com/bueler/p4pdes.
The discretization is by Q_1 finite elements.  This code is multigrid (GMG)
The prefix for PETSC solver options is 's_'.  Use -help for PETSc options
and -minhelp for options to this program.""",
    formatter_class=RawTextHelpFormatter,add_help=False)
parser.add_argument('-minhelp', action='store_true', default=False,
                    help='help for minimal.py options')
parser.add_argument('-mx', type=int, default=3, metavar='MX',
                    help='number of grid points in x-direction')
parser.add_argument('-my', type=int, default=3, metavar='MY',
                    help='number of grid points in y-direction')
parser.add_argument('-o', metavar='NAME', type=str, default='',
                    help='output file name ending with .pvd')
parser.add_argument('-k', type=int, default=1, metavar='K',
                    help='polynomial degree for Q_k elements')
parser.add_argument('-refine', type=int, default=-1, metavar='X',
                    help='number of refinement levels (e.g. for GMG)')
args, unknown = parser.parse_known_args()
if args.minhelp:
    parser.print_help()

# Create mesh, enabling GMG via refinement using hierarchy
mx, my = args.mx, args.my
mesh = UnitSquareMesh(mx-1, my-1, quadrilateral=True)
if args.refine > 0:
    hierarchy = MeshHierarchy(mesh, args.refine)
    mesh = hierarchy[-1]     # the fine mesh
    mx, my = (mx-1) * 2**args.refine + 1, (my-1) * 2**args.refine + 1
x,y = SpatialCoordinate(mesh)
mesh._plex.viewFromOptions('-dm_view')
# to print coordinates:  print(mesh.coordinates.dat.data)

# Define function space, right-hand side, and weak form.
W = FunctionSpace(mesh, 'Lagrange', degree=args.k)
FIXME
f_rhs = Function(W).interpolate(x * exp(y))  # manufactured
u = Function(W)  # initialized to zero here
v = TestFunction(W)
F = (dot(grad(u), grad(v)) - f_rhs * v) * dx

# Define Dirichlet boundary conditions
g_bdry = Function(W).interpolate(- x * exp(y))  # = exact solution
bdry_ids = (1, 2, 3, 4)   # all four sides of boundary
bc = DirichletBC(W, g_bdry, bdry_ids)

# Solve system as though it is nonlinear:  F(u) = 0
solve(F == 0, u, bcs = [bc], options_prefix = 's',
      solver_parameters = {'snes_type': 'ksponly',
                           'ksp_type': 'cg'})

# Print numerical error in L_infty and L_2 norm
elementstr = '%s^%d' % (['P','Q'][args.quad],args.k)
udiff = Function(W).interpolate(u - g_bdry)
with udiff.dat.vec_ro as vudiff:
    error_Linf = abs(vudiff).max()[1]
error_L2 = sqrt(assemble(dot(udiff, udiff) * dx))
PETSc.Sys.Print('done on %d x %d grid with %s elements:' \
      % (mx,my,elementstr))
PETSc.Sys.Print('  error |u-uexact|_inf = %.3e, |u-uexact|_h = %.3e' \
      % (error_Linf,error_L2))

# Optionally save to a .pvd file viewable with Paraview
if len(args.o) > 0:
    PETSc.Sys.Print('saving solution to %s ...' % args.o)
    u.rename('u')
    File(args.o).write(u)

