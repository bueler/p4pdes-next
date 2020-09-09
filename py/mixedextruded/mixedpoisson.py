#!/usr/bin/env python3

# from section 3.1 of Gibson et al 2019

from firedrake import *

m = 16
mesh = UnitSquareMesh(m,m)

U = FunctionSpace(mesh, "RT", 1)
V = FunctionSpace(mesh, "DG", 0)

W = U * V

sigma, u = TrialFunctions(W)
tau, v = TestFunctions(W)

f = Function(V)
x,y = SpatialCoordinate(mesh)
f.interpolate(-2*(x-1)*x - 2*(y-1)*y)

a = (dot(tau, sigma) - u * div(tau) + div(sigma) * v) * dx
L = f*v*dx

w = Function(W, name="Solution")
solve(a == L, w)

sigmasoln, usoln = w.split()
sigmasoln.rename('sigma')
usoln.rename('u')
File("output.pvd").write(sigmasoln,usoln)

