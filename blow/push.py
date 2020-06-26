from firedrake.petsc import PETSc
foo = PETSc.Viewer(PETSc.Viewer.Type.ASCII).create()
foo.pushASCIITab()
foo.printfASCII('foo')

