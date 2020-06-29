from firedrake.petsc import PETSc
foo = PETSc.Viewer().STDOUT()
foo.printfASCII('foo\n')
foo.pushASCIITab()
foo.printfASCII('foo\n')

