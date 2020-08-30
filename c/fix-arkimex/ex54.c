static char help[] =
"Simple ODE system to test implicit and IMEX methods when both RHSFunction()\n"
"and IFunction() are supplied, with Jacobians as well.  Problem has form\n"
"   F(t,u,dudt) = G(t,u)\n"
"where  u(t) = (x(t),y(t))  and x,y are scalar.  Functions F (IFunction) and\n"
"G (RHSFunction) are actually linear, but this example uses TS_NONLINEAR type.\n"
"In fact the system is\n"
"   x' + y' = 6 y\n"
"        y' = x\n"
"with x(0)=1 and y(0)=3.  The exact solution is known; see end of main()\n"
"for computation of numerical error.  Defaults to BDF TS type.\n\n";

/* with BDF2, BDF2+-snes_fd, BDF6+tight tol., CN, ROSW:
$ ./ex54
error norm at tf = 1.000000 from 33 steps:  |u-u_exact| =  9.29170e-02
$ ./ex54 -snes_fd
error norm at tf = 1.000000 from 33 steps:  |u-u_exact| =  9.29170e-02
$ ./ex54 -ts_rtol 1.0e-14 -ts_atol 1.0e-14 -ts_bdf_order 6
error norm at tf = 1.000000 from 388 steps:  |u-u_exact| =  4.23624e-11
$ ./ex54 -ts_type cn
error norm at tf = 1.000000 from 100 steps:  |u-u_exact| =  2.22839e-03
$ ./ex54 -ts_type rosw
error norm at tf = 1.000000 from 21 steps:  |u-u_exact| =  5.64012e-03

BROKEN with arkimex:
$ ./ex54 -ts_type arkimex
error norm at tf = 1.000000 from 16 steps:  |u-u_exact| =  1.93229e+01
*/

#include <petsc.h>

extern PetscErrorCode FormIFunction(TS,PetscReal,Vec,Vec,Vec F,void*);
extern PetscErrorCode FormIJacobian(TS,PetscReal,Vec,Vec,PetscReal,Mat,Mat,void*);
extern PetscErrorCode FormRHSFunction(TS,PetscReal,Vec,Vec,void*);
extern PetscErrorCode FormRHSJacobian(TS,PetscReal,Vec,Mat,Mat,void*);

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  TS             ts;
  Vec            u, uexact;
  Mat            JF,JG;
  PetscReal      tf,xf,yf,errnorm;
  PetscInt       steps;

  PetscInitialize(&argc,&argv,(char*)0,help);

  ierr = VecCreate(PETSC_COMM_WORLD,&u); CHKERRQ(ierr);
  ierr = VecSetSizes(u,PETSC_DECIDE,2); CHKERRQ(ierr);
  ierr = VecSetFromOptions(u); CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_WORLD,&JF); CHKERRQ(ierr);
  ierr = MatSetSizes(JF,PETSC_DECIDE,PETSC_DECIDE,2,2); CHKERRQ(ierr);
  ierr = MatSetFromOptions(JF); CHKERRQ(ierr);
  ierr = MatSetUp(JF); CHKERRQ(ierr);
  ierr = MatDuplicate(JF,MAT_DO_NOT_COPY_VALUES,&JG); CHKERRQ(ierr);

  ierr = TSCreate(PETSC_COMM_WORLD,&ts); CHKERRQ(ierr);

  ierr = TSSetIFunction(ts,NULL,FormIFunction,NULL); CHKERRQ(ierr);
  ierr = TSSetIJacobian(ts,JF,JF,FormIJacobian,NULL); CHKERRQ(ierr);

  ierr = TSSetRHSFunction(ts,NULL,FormRHSFunction,NULL); CHKERRQ(ierr);
  ierr = TSSetRHSJacobian(ts,JG,JG,FormRHSJacobian,NULL); CHKERRQ(ierr);

  ierr = TSSetProblemType(ts,TS_NONLINEAR); CHKERRQ(ierr);
  ierr = TSSetType(ts,TSBDF); CHKERRQ(ierr);
  ierr = TSSetTime(ts,0.0); CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts,1.0); CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,0.01); CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP); CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  ierr = VecSetValue(u,0,1.0,INSERT_VALUES); CHKERRQ(ierr);
  ierr = VecSetValue(u,1,3.0,INSERT_VALUES); CHKERRQ(ierr);
  ierr = VecAssemblyBegin(u); CHKERRQ(ierr);
  ierr = VecAssemblyEnd(u); CHKERRQ(ierr);
  ierr = TSSolve(ts,u); CHKERRQ(ierr);

  ierr = TSGetStepNumber(ts,&steps); CHKERRQ(ierr);
  ierr = TSGetTime(ts,&tf); CHKERRQ(ierr);
  xf = -3.0 * PetscExpReal(-3.0*tf) + 4.0 * PetscExpReal(2.0*tf);
  yf = PetscExpReal(-3.0*tf) + 2.0 * PetscExpReal(2.0*tf);
  //ierr = PetscPrintf(PETSC_COMM_WORLD,
  //         "xf = %.6f, yf = %.6f\n",xf,yf); CHKERRQ(ierr);
  ierr = VecDuplicate(u,&uexact); CHKERRQ(ierr);
  ierr = VecSetValue(uexact,0,xf,INSERT_VALUES); CHKERRQ(ierr);
  ierr = VecSetValue(uexact,1,yf,INSERT_VALUES); CHKERRQ(ierr);
  ierr = VecAssemblyBegin(uexact); CHKERRQ(ierr);
  ierr = VecAssemblyEnd(uexact); CHKERRQ(ierr);
  ierr = VecAXPY(u,-1.0,uexact); CHKERRQ(ierr); // u <- u + (-1.0) uexact
  ierr = VecNorm(u,NORM_2,&errnorm); CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,
           "error norm at tf = %.6f from %d steps:  |u-u_exact| =  %.5e\n",
           tf,steps,errnorm); CHKERRQ(ierr);

  VecDestroy(&u);  VecDestroy(&uexact);
  MatDestroy(&JF);  MatDestroy(&JG);
  TSDestroy(&ts);
  return PetscFinalize();
}


PetscErrorCode FormIFunction(TS ts, PetscReal t, Vec u, Vec dudt,
                             Vec F, void *ctx) {
    PetscErrorCode  ierr;
    const PetscReal *au, *adudt;
    PetscReal       *aF;

    ierr = VecGetArrayRead(u,&au); CHKERRQ(ierr);
    ierr = VecGetArrayRead(dudt,&adudt); CHKERRQ(ierr);
    ierr = VecGetArray(F,&aF);
    aF[0] = adudt[0] + adudt[1];
    aF[1] =            adudt[1];
    ierr = VecRestoreArrayRead(u,&au); CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(dudt,&adudt); CHKERRQ(ierr);
    ierr = VecRestoreArray(F,&aF); CHKERRQ(ierr);
    return 0;
}

// computes  J = dF/du + a dF/d(dudt)
PetscErrorCode FormIJacobian(TS ts, PetscReal t, Vec u, Vec dudt,
                             PetscReal a, Mat J, Mat P, void *ctx) {
    PetscErrorCode ierr;
    PetscInt       row[2] = {0, 1},  col[2] = {0, 1};
    PetscReal      v[4] = { a,   a,
                            0.0, a};
    ierr = MatSetValues(P,2,row,2,col,v,INSERT_VALUES); CHKERRQ(ierr);
    ierr = MatAssemblyBegin(P,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(P,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    if (J != P) {
        ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
        ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    }
    return 0;
}

PetscErrorCode FormRHSFunction(TS ts, PetscReal t, Vec u,
                               Vec G, void *ctx) {
    PetscErrorCode  ierr;
    const PetscReal *au;
    PetscReal       *aG;

    ierr = VecGetArrayRead(u,&au); CHKERRQ(ierr);
    ierr = VecGetArray(G,&aG); CHKERRQ(ierr);
    aG[0] = 6.0 * au[1];
    aG[1] = au[0];
    ierr = VecRestoreArrayRead(u,&au); CHKERRQ(ierr);
    ierr = VecRestoreArray(G,&aG); CHKERRQ(ierr);
    return 0;
}

PetscErrorCode FormRHSJacobian(TS ts, PetscReal t, Vec u, Mat J, Mat P,
                               void *ctx) {
    PetscErrorCode ierr;
    PetscInt       row[2] = {0, 1},  col[2] = {0, 1};
    PetscReal      v[4] = { 0.0, 6.0,
                            1.0, 0.0};
    ierr = MatSetValues(P,2,row,2,col,v,INSERT_VALUES); CHKERRQ(ierr);
    ierr = MatAssemblyBegin(P,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(P,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    if (J != P) {
        ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
        ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    }
    return 0;
}

