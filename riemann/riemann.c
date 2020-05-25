static char help[] =
"Solve hyperbolic system in one space dimension (1D):\n"
"    u_t + (A(t,x) u)_x = g(t,x,u)\n"  // FIXME implement nonlinear A(t,x,u)
"where u(t,x) is column vector of length n.  Uses finite volumes and\n"
"a Riemann solver at cell faces to compute the flux.  Here A(t,x) is an\n"
"n x n matrix and g(t,x,u) is a column vector of length n.  Domain is (t,x) in\n"
"[0,T]x[a,b].  Initial condition is u(0,x) = f(x) and boundary conditions are\n"
"given flux:  A(t,0) u(t,0) = PhiL(t),  A(t,L) u(t,L) = PhiR(t).\n\n";

// at this stage, following works:
//   ./riemann -da_grid_x 100 -ts_monitor -ts_monitor_solution draw -draw_pause 0.1

#include <petsc.h>

/* The struct "ProblemCtx" is defined in cases.h, including these problems:
   acoustic
Functions CreateCase(), DestroyCase() are also defined in cases.h, and called
in main() below. */
#include "cases.h"

extern PetscErrorCode FormInitial(DMDALocalInfo*, Vec, PetscReal, ProblemCtx*);
extern PetscErrorCode FormRHSFunctionLocal(DMDALocalInfo*, PetscReal,
        PetscReal*, PetscReal*, void*);

int main(int argc,char **argv) {
    PetscErrorCode   ierr;
    TS               ts;                 // ODE solver for method-of-lines (MOL)
    DM               da;                 // structured grid
    Vec              u;                  // the solution
    DMDALocalInfo    info;               // structured grid info
    ProblemCtx       user;               // problem-specific information
    char             fieldnamestr[20];// FIXME use fieldnames from ProblemCtx
    PetscInt         k, steps;
    PetscReal        hx, t0=0.0, dt=0.1, tf=1.0; // FIXME control by options

    PetscInitialize(&argc,&argv,(char*)0,help);
    ierr = CreateCase(0,&user); CHKERRQ(ierr);

    // create grid
    ierr = DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,4,user.n_dim,1,NULL,&da); CHKERRQ(ierr);
    ierr = DMSetFromOptions(da); CHKERRQ(ierr);
    ierr = DMSetUp(da); CHKERRQ(ierr);
    ierr = DMSetApplicationContext(da,&user); CHKERRQ(ierr);

    // set field names so that visualization makes sense
    ierr = DMDAGetLocalInfo(da,&info); CHKERRQ(ierr);
    for (k = 0; k < info.dof; k++) {
        snprintf(fieldnamestr,19,"u%d",k);
        ierr = DMDASetFieldName(da,k,fieldnamestr); CHKERRQ(ierr);
    }

    // create TS:  du/dt = G(t,u)  form
    ierr = TSCreate(PETSC_COMM_WORLD,&ts); CHKERRQ(ierr);
    ierr = TSSetProblemType(ts,TS_NONLINEAR); CHKERRQ(ierr);
    ierr = TSSetDM(ts,da); CHKERRQ(ierr);
    ierr = DMDATSSetRHSFunctionLocal(da,INSERT_VALUES,
           (DMDATSRHSFunctionLocal)FormRHSFunctionLocal,&user); CHKERRQ(ierr);
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
    ierr = FormInitial(&info,u,t0,&user); CHKERRQ(ierr);
    //ierr = VecView(u,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);

    // solve
    hx = (user.b_right - user.a_left) / info.mx;
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


PetscErrorCode FormInitial(DMDALocalInfo *info, Vec u, PetscReal t0, ProblemCtx *user) {
    PetscErrorCode   ierr;
    const PetscReal  hx = (user->b_right - user->a_left) / info->mx;
    PetscInt         i;
    PetscReal        x, *au;

    ierr = DMDAVecGetArray(info->da, u, &au); CHKERRQ(ierr);
    for (i=info->xs; i<info->xs+info->xm; i++) {
        x = user->a_left + (i+0.5) * hx;
        ierr = user->f_initial(t0,x,&au[(info->dof)*i]); CHKERRQ(ierr);
    }
    ierr = DMDAVecRestoreArray(info->da, u, &au); CHKERRQ(ierr);
    return 0;
}


// FIXME implement better boundary conditions
// FIXME implement slope limiters
// Right-hand-side of method-of-lines discretization of PDE.  Implements
// Gudonov (Riemann-solver upwind) method.
PetscErrorCode FormRHSFunctionLocal(DMDALocalInfo *info, PetscReal t,
        PetscReal *au, PetscReal *aG, void *ctx) {
    PetscErrorCode   ierr;
    ProblemCtx       *user = (ProblemCtx*)ctx;
    const PetscReal  hx = (user->b_right - user->a_left) / info->mx;
    PetscInt         i, k;
    PetscReal        x, *Fl, *Fr;     // Fl,Fr hold current cell face fluxes

    //ierr = PetscPrintf(PETSC_COMM_SELF,
    //    "xs=%d,xm=%d,mx=%d,dof=%d\n",info->xs,info->xm,info->mx,info->dof); CHKERRQ(ierr);
    ierr = PetscMalloc1(info->dof,&Fl); CHKERRQ(ierr);
    ierr = PetscMalloc1(info->dof,&Fr); CHKERRQ(ierr);

    // get flux at left boundary (of first cell owned by process)
    if (info->xs == 0) {
        ierr = user->PhiL_bdryflux(t,&au[(info->dof)*(info->xs)],Fl); CHKERRQ(ierr);
    } else {
        x = user->a_left + (info->xs+0.5) * hx;
        ierr = user->faceflux(t,x-hx/2.0,&au[(info->dof)*(info->xs-1)],
                                         &au[(info->dof)*(info->xs)],Fl); CHKERRQ(ierr);
    }

    // i counts cell center
    for (i = info->xs; i < info->xs + info->xm; i++) {
        x = user->a_left + (i+0.5) * hx;
        // aG[n i + k]  filled from g(t,x,u) first
        ierr = user->g_source(t,x,&au[(info->dof)*i],&aG[(info->dof)*i]); CHKERRQ(ierr);
        // get right-face flux for cell
        if (i == info->mx-1) {
            ierr = user->PhiR_bdryflux(t,&au[(info->dof)*i],Fr); CHKERRQ(ierr);
        } else {
            ierr = user->faceflux(t,x+hx/2.0,&au[(info->dof)*i],
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

