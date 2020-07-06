Differential geometry of surfaces
=================================

This is not the usual story of differential geometry.  It is instead
an idiot's guide, with Firedrake as a crutch to stumble through some
calculus of surfaces, especially minimal surfaces, eventually reaching
a better understanding and an efficient finite element formulation.

We work with parameterized surfaces in :math:`\mathbb{R}^3`,

.. math::

  X(s,t) = (x(s,t),y(s,t),z(s,t))

where :math:`x,y,z` are scalar functions.  Assume
:math:`(s,t)\in \Omega \subset \mathbb{R}^2`.  Most calculus books
have a simple formula for the area of a surface, namely

.. math::

  A(X) = \int_{\quad\Omega} \|X_s \times X_t\| \,\mathrm{d} s \mathrm{d} t

where :math:`X_s,X_t` denote partial derivatives of the vector-valued
function :math:`X` and the norm :math:`\|\cdot\|` is the usual (Euclidean)
vector magnitude in :math:`\mathbb{R}^3`.

This demonstration shows how to solve for, and plot, parameterized surfaces
which are extremals of this area functional, namely minimal surfaces.
We will solve the Dirichlet problem, a.k.a. Plateau's problem or the soap
bubble problem:

.. math::

  X\big|_{\partial \Omega} = G

where :math:`G(s,t)` is a known function.

To find a minimum we take a derivative and set it to zero.  In this case
we compute the first variation of :math:`A(X)`.  

  

.. math::

  \|X_s \times X_t\| = \left[\left(X_s\times X_t\right)\cdot  \left(X_s\times X_t\right)\right]^{1/2}

We set up a mesh on :math:`(s,t)` space in the usual way. ::

  from firedrake import *
  mesh = UnitSquareMesh(9,9)

# Enable GMG by refinement hierarchy, and grid-sequencing by further refinement
if args.refine + args.sequence > 0:
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

.. FIXME: add a minimal surface that is not so boring, with x(s,t), y(s,t) less
          trivial

