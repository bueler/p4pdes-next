static char help[] =
"Solve hyperbolic system in 1D:  u_t + (A u)_x = b  where u(t,x) is length n.\n"
"Domain is (t,x) in [0,T]x[0,L].  Here A(t,x) is an n x n matrix and b(t,x,u)\n"
"returns an n-length vector.  Initial condition u(0,x) = g(x) and boundary flux\n"
"conditions  A(t,0) u(t,0) = Fl(t),  A(t,L) u(t,L) = Fr(t),  for g, Fl, Fr given.\n\n";

// at this stage, following "works":
//   ./riemann -da_grid_x 20 -ts_monitor -ts_monitor_solution draw -draw_pause 0.2

#include <petsc.h>

// ********** START wave equation **********
const PetscInt  usern = 2;
const PetscReal userT = 1.0,
                userL = 1.0;

PetscErrorCode userb(PetscReal t, PetscReal x, PetscReal *u) {
    PetscInt l;
    for (l = 0; l < usern; l++)
        u[l] = 0.0;
    return 0;
}

PetscErrorCode userg(PetscReal t, PetscReal x, PetscReal *u) {
    if (x > 0.4 && x < 0.6)
        u[0] = 1.0;
    else
        u[0] = 0.0;
    u[1] = 0.0;
    return 0;
}

PetscErrorCode userfluxleft(PetscReal t, PetscReal *F) {
    PetscInt l;
    for (l = 0; l < usern; l++)
        F[l] = 0.0;
    return 0;
}

PetscErrorCode userfluxright(PetscReal t, PetscReal *F) {
    PetscInt l;
    for (l = 0; l < usern; l++)
        F[l] = 0.0;
    return 0;
}

// here is where all the action is ... needs clear documentation
PetscErrorCode userfaceflux(PetscReal t, PetscReal x,
       PetscReal *ul, PetscReal *ur, PetscReal *F, PetscReal *speed) {
    PetscReal      c = 1.0,
                   c0 = (ul[0] + ul[1]) / 2.0,
                   c1 = (ur[0] - ur[1]) / 2.0;
    // uface[0] = c0 + c1;
    // uface[1] = c0 - c1;
    // F = A uface
    F[0] = c * (c0 - c1);
    F[1] = c * (c0 + c1);
    if (speed != NULL)
        *speed = c;
    return 0;
}
// ********** END wave equation **********


extern PetscErrorCode FormInitial(DMDALocalInfo*, Vec, PetscReal);
extern PetscErrorCode FormRHSFunctionLocal(DMDALocalInfo*, PetscReal,
        PetscReal*, PetscReal*, void*);

int main(int argc,char **argv) {
    PetscErrorCode   ierr;
    TS               ts;
    DM               da;
    Vec              u;
    DMDALocalInfo    info;
    PetscInt         steps;
    PetscReal        hx, t0=0.0, dt=0.1, tf=userT, L=userL;

    PetscInitialize(&argc,&argv,(char*)0,help);

    ierr = DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,4,usern,1,NULL,&da); CHKERRQ(ierr);
    ierr = DMSetFromOptions(da); CHKERRQ(ierr);
    ierr = DMSetUp(da); CHKERRQ(ierr);
    //ierr = DMSetApplicationContext(da,&user); CHKERRQ(ierr);
    ierr = DMDASetFieldName(da,0,"u"); CHKERRQ(ierr);
    ierr = DMDASetFieldName(da,1,"v"); CHKERRQ(ierr);

    ierr = TSCreate(PETSC_COMM_WORLD,&ts); CHKERRQ(ierr);
    ierr = TSSetProblemType(ts,TS_NONLINEAR); CHKERRQ(ierr);
    ierr = TSSetDM(ts,da); CHKERRQ(ierr);
    ierr = DMDATSSetRHSFunctionLocal(da,INSERT_VALUES,
           (DMDATSRHSFunctionLocal)FormRHSFunctionLocal,NULL); CHKERRQ(ierr);
    ierr = TSSetType(ts,TSRK); CHKERRQ(ierr);  // defaults to -ts_rk_type 3bs

    // time axis
    ierr = TSSetTime(ts,t0); CHKERRQ(ierr);
    ierr = TSSetMaxTime(ts,tf); CHKERRQ(ierr);
    ierr = TSSetTimeStep(ts,dt); CHKERRQ(ierr);
    ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP); CHKERRQ(ierr);
    ierr = TSSetFromOptions(ts);CHKERRQ(ierr);
    ierr = TSGetTime(ts,&t0); CHKERRQ(ierr);
    ierr = TSGetTimeStep(ts,&dt); CHKERRQ(ierr);

    ierr = DMCreateGlobalVector(da,&u); CHKERRQ(ierr);
    ierr = DMDAGetLocalInfo(da,&info); CHKERRQ(ierr);
    ierr = FormInitial(&info,u,t0); CHKERRQ(ierr);
    //ierr = VecView(u,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);

    hx = L / info.mx;
    ierr = PetscPrintf(PETSC_COMM_WORLD,
               "solving system of %d equations on %d-point grid with dx=%g ...\n",
               info.dof,info.mx,hx); CHKERRQ(ierr);

    ierr = TSSolve(ts,u); CHKERRQ(ierr);

    ierr = TSGetStepNumber(ts,&steps); CHKERRQ(ierr);
    ierr = TSGetTime(ts,&tf); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,
               "completed %d steps to time %g\n",steps,tf); CHKERRQ(ierr);

    VecDestroy(&u);  TSDestroy(&ts);  DMDestroy(&da);
    return PetscFinalize();
}


PetscErrorCode FormInitial(DMDALocalInfo *info, Vec u, PetscReal t0) {
    PetscErrorCode ierr;
    PetscInt   i;
    PetscReal  hx, x, *au;

    ierr = DMDAVecGetArray(info->da, u, &au); CHKERRQ(ierr);
    hx = userL / info->mx;
    for (i=info->xs; i<info->xs+info->xm; i++) {
        x = 0.0 + (i+0.5) * hx;
        ierr = userg(t0,x,&au[(info->dof)*i]); CHKERRQ(ierr);
    }
    ierr = DMDAVecRestoreArray(info->da, u, &au); CHKERRQ(ierr);
    return 0;
}


// FIXME
// method-of-lines discretization
PetscErrorCode FormRHSFunctionLocal(DMDALocalInfo *info, PetscReal t,
        PetscReal *au, PetscReal *aG, void *user) {
    PetscInt   i, l;

    // clear G first
    for (i = info->xs; i < info->xs + info->xm; i++)
        for (l = 0; l < info->dof; l++)
            aG[(info->dof)*i+l] = 0.0;
    return 0;
}

