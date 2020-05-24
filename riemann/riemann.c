static char help[] =
"Solve hyperbolic system in 1D:\n"
"    u_t + (A(t,x) u)_x = b(t,x,u)\n"
"where u(t,x) is column vector of length n.  Uses finite volume thinking and\n"
"a Riemann solver at cell faces to compute the flux.  Here A(t,x) is an\n"
"n x n matrix and b(t,x,u) is a column vector of length n.  Domain is (t,x) in\n"
"[0,T]x[0,L].  Initial condition is u(0,x) = g(x) and boundary conditions are\n"
"given flux:  A(t,0) u(t,0) = Fl(t),  A(t,L) u(t,L) = Fr(t).\n\n";

// at this stage, following works:
//   ./riemann -da_grid_x 100 -ts_monitor -ts_monitor_solution draw -draw_pause 0.1

#include <petsc.h>

// ********** START wave equation **********
const PetscInt  usern = 2;
const PetscReal userT = 0.25,
                userL = 1.0;

PetscErrorCode userb(PetscReal t, PetscReal x, PetscReal *u, PetscReal *b) {
    PetscInt l;
    for (l = 0; l < usern; l++)
        b[l] = 0.0;
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
       PetscReal *ul, PetscReal *ur, PetscReal *F) {
    PetscReal      c = 1.0,
                   c0 = (ul[0] + ul[1]) / 2.0,
                   c1 = (ur[0] - ur[1]) / 2.0;
    // uface[0] = c0 + c1;
    // uface[1] = c0 - c1;
    // F = A uface
    F[0] = c * (c0 - c1);
    F[1] = c * (c0 + c1);
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
    char             namestr[20];
    PetscInt         k, steps;
    PetscReal        hx, t0=0.0, dt=0.1, tf=userT, L=userL;

    PetscInitialize(&argc,&argv,(char*)0,help);

    // create grid
    ierr = DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,4,usern,1,NULL,&da); CHKERRQ(ierr);
    ierr = DMSetFromOptions(da); CHKERRQ(ierr);
    ierr = DMSetUp(da); CHKERRQ(ierr);
    //ierr = DMSetApplicationContext(da,&user); CHKERRQ(ierr);

    // set field names so that visualization makes sense
    ierr = DMDAGetLocalInfo(da,&info); CHKERRQ(ierr);
    for (k = 0; k < info.dof; k++) {
        snprintf(namestr,19,"u%d",k);
        ierr = DMDASetFieldName(da,k,namestr); CHKERRQ(ierr);
    }

    // create TS:  du/dt = G(t,u)  form
    ierr = TSCreate(PETSC_COMM_WORLD,&ts); CHKERRQ(ierr);
    ierr = TSSetProblemType(ts,TS_NONLINEAR); CHKERRQ(ierr);
    ierr = TSSetDM(ts,da); CHKERRQ(ierr);
    ierr = DMDATSSetRHSFunctionLocal(da,INSERT_VALUES,
           (DMDATSRHSFunctionLocal)FormRHSFunctionLocal,NULL); CHKERRQ(ierr);
    ierr = TSSetType(ts,TSRK); CHKERRQ(ierr);  // defaults to -ts_rk_type 3bs

    // set up time axis
    ierr = TSSetTime(ts,t0); CHKERRQ(ierr);
    ierr = TSSetMaxTime(ts,tf); CHKERRQ(ierr);
    ierr = TSSetTimeStep(ts,dt); CHKERRQ(ierr);
    ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP); CHKERRQ(ierr);
    ierr = TSSetFromOptions(ts);CHKERRQ(ierr);
    ierr = TSGetTime(ts,&t0); CHKERRQ(ierr);
    ierr = TSGetTimeStep(ts,&dt); CHKERRQ(ierr);

    // get initial values
    ierr = DMCreateGlobalVector(da,&u); CHKERRQ(ierr);
    ierr = FormInitial(&info,u,t0); CHKERRQ(ierr);
    //ierr = VecView(u,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);

    // solve
    hx = L / info.mx;
    ierr = PetscPrintf(PETSC_COMM_WORLD,
               "solving system of %d equations on %d-point grid with dx=%g ...\n",
               info.dof,info.mx,hx); CHKERRQ(ierr);
    ierr = TSSolve(ts,u); CHKERRQ(ierr);

    // report on solution
    ierr = TSGetStepNumber(ts,&steps); CHKERRQ(ierr);
    ierr = TSGetTime(ts,&tf); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,
               "... completed %d steps to time %g\n",steps,tf); CHKERRQ(ierr);

    VecDestroy(&u);  TSDestroy(&ts);  DMDestroy(&da);
    return PetscFinalize();
}


PetscErrorCode FormInitial(DMDALocalInfo *info, Vec u, PetscReal t0) {
    PetscErrorCode ierr;
    PetscInt   i;
    PetscReal  hx = userL / info->mx, x, *au;

    ierr = DMDAVecGetArray(info->da, u, &au); CHKERRQ(ierr);
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
        PetscReal *au, PetscReal *aG, void *tmp) {
    PetscErrorCode ierr;
    PetscInt   i, k;
    PetscReal  hx = userL / info->mx, x, *Fl, *Fr;

    //ierr = PetscPrintf(PETSC_COMM_SELF,
    //    "xs=%d,xm=%d,mx=%d,dof=%d\n",info->xs,info->xm,info->mx,info->dof); CHKERRQ(ierr);

    ierr = PetscMalloc1(info->dof,&Fl); CHKERRQ(ierr);
    ierr = PetscMalloc1(info->dof,&Fr); CHKERRQ(ierr);

    // get flux at left boundary (of first cell owned by process)
    if (info->xs == 0) {
        ierr = userfluxleft(t,Fl); CHKERRQ(ierr);
    } else {
        x = 0.0 + (info->xs+0.5) * hx;
        ierr = userfaceflux(t,x-hx/2.0,&au[(info->dof)*(info->xs-1)],
                                       &au[(info->dof)*(info->xs)],Fl); CHKERRQ(ierr);
    }

    // i counts cell center
    for (i = info->xs; i < info->xs + info->xm; i++) {
        x = 0.0 + (i+0.5) * hx;
        // aG[n i + k]  filled from b(t,x,u) first
        ierr = userb(t,x,&au[(info->dof)*i],&aG[(info->dof)*i]); CHKERRQ(ierr);
        // get right-face flux for cell
        if (i == info->mx-1) {
            ierr = userfluxright(t,Fr); CHKERRQ(ierr);
        } else {
            ierr = userfaceflux(t,x+hx/2.0,&au[(info->dof)*i],
                                           &au[(info->dof)*(i+1)],Fr); CHKERRQ(ierr);
        }
        // compute RHS
        for (k = 0; k < info->dof; k++)
            aG[(info->dof)*i+k] += (Fl[k] - Fr[k]) / hx;
        // transfer Fr to Fl
        for (k = 0; k < info->dof; k++)
            Fl[k] = Fr[k];
    }

    ierr = PetscFree(Fl); CHKERRQ(ierr);
    ierr = PetscFree(Fr); CHKERRQ(ierr);
    return 0;
}

