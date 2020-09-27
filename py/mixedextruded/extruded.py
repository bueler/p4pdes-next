#!/usr/bin/env python3

# from section 3.2 and 3.3 of Gibson et al 2019: construction of extruded meshes

from firedrake import *

# cubed sphere  (with more-visualizable geometric settings)

#base_mesh = CubedSphereMesh(radius=6400.0e6, refinement_level=5, degree=2)
#mesh = ExtrudedMesh(base_mesh, layers=20, layer_height=10,
#                    extrusion_type="radial")
base_mesh = CubedSphereMesh(radius=200.0, refinement_level=2, degree=2)
mesh = ExtrudedMesh(base_mesh, layers=5, layer_height=10,
                    extrusion_type="radial")
V = FunctionSpace(mesh, "CG", 1)
u = Function(V)
File("cubedsphere.pvd").write(u)


# unit square as extruded interval, including H(curl) and H(div) spaces

#N = 20
N = 7
base_mesh = UnitIntervalMesh(N)
mesh = ExtrudedMesh(base_mesh, layers=N, layer_height=1.0/N)
P2 = FiniteElement("CG", interval, 2)
P3 = FiniteElement("CG", interval, 3)
H1_element = TensorProductElement(P2, P3)
H1 = FunctionSpace(mesh, H1_element)
dP1 = FiniteElement("DG", interval, 1)
dP2 = FiniteElement("DG", interval, 2)
L2_element = TensorProductElement(dP1, dP2)
L2 = FunctionSpace(mesh, L2_element)
S = TensorProductElement(P2, dP1)
Hcurl = FunctionSpace(mesh, HCurl(S))
Hdiv = FunctionSpace(mesh, HDiv(S))
x, y = SpatialCoordinate(mesh)
uH1 = Function(H1).interpolate(2*x*sin(4*pi*x) + y*y*y*y)
uH1.rename('uH1')
uL2 = Function(L2).interpolate(2*x*sin(4*pi*x) + y*y*y*y)
uL2.rename('uL2')
uHcurl = Function(Hcurl).project(as_vector([y*y,x*x]))  # interpolate fails
uHcurl.rename('uHcurl')
uHdiv = Function(Hdiv).project(as_vector([y*y,x*x]))  # interpolate fails
uHdiv.rename('uHdiv')
File("unitsquare.pvd").write(uH1,uL2,uHcurl,uHdiv)


# unit cube as extruded unit square, including H(curl) and H(div) spaces

#N = 10
N = 5
base_mesh = UnitSquareMesh(N, N)
mesh = ExtrudedMesh(base_mesh, layers=N, layer_height=1.0/N)
P2t = FiniteElement("CG", triangle, 2)
P2i = FiniteElement("CG", interval, 2)
H1_element = TensorProductElement(P2t, P2i)
H1 = FunctionSpace(mesh, H1_element)
dP1t = FiniteElement("DG", triangle, 1)
dP1i = FiniteElement("DG", interval, 1)
L2_element = TensorProductElement(dP1t, dP1i)
L2 = FunctionSpace(mesh, L2_element)
N2_1 = FiniteElement("N2curl", triangle, 1)
Hcurl_h = HCurl(TensorProductElement(N2_1, P2i))
Hcurl_v = HCurl(TensorProductElement(P2t,dP1i))

# FIXME

