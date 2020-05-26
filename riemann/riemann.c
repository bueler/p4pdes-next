static char help[] =
"Solve a hyperbolic system in one space dimension (1D):\n"
"    q_t + F(t,x,q)_x = g(t,x,q)\n"
"where solution q(t,x), flux F(t,x,q), and source g(t,x,q) are column vectors\n"
"of length n.  Option prefix rie_.  The domain is (t,x) in [0,T]x[a,b].\n"
"The initial condition is q(0,x) = f(x).\n"
"  The flux may be of the form\n"
"F(t,x,q) = A(t,x,q) q  (but this is not actually required).  Uses finite\n"
"volumes and a Riemann solver\n"
"    F = faceflux(t,x,qleft,qright)\n"
"at cell faces.  That is, Godunov's method is used: at each cell face use the\n"
"Riemann solver to compute the value on the cell face going forward in time.\n"
"Similarly, it uses flux boundary conditions\n"
"    F = bdryflux_a(t,qright),\n"
"    F = bdryflux_b(t,qleft),\n"
"which allows at least reflecting and outflow boundary conditions.\n"
"  Use -problem X to select a problem from:\n"
"    acoustic   classical wave equation in n=2 system form\n"
"    swater     shallow water equations (n=2)\n"
"Defaults to acoustic.\n"
"  Use options like these for further information and control\n"
"    -da_grid_x\n"
"    -ts_monitor\n"
"    -ts_monitor_solution draw -draw_pause 0.1\n"
"    -ts_type                     [default is rk]\n"
"      -ts_rk_type                [default is 3bs]\n"
"See the makefile for test examples, and do 'make test' to test.\n";


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
    Vec              q;                  // the solution
    DMDALocalInfo    info;               // structured grid info
    ProblemType      problem;            // which problem we are solving
    ProblemCtx       user;               // problem-specific information
    PetscInt         k, steps;
    PetscReal        hx, qnorm, t0, tf;

    PetscInitialize(&argc,&argv,(char*)0,help);
    ierr = CreateCase(0,&problem,&user); CHKERRQ(ierr);

    // create grid
    ierr = DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,4,
                        user.n_dim,1,NULL,&da); CHKERRQ(ierr);
    ierr = DMSetFromOptions(da); CHKERRQ(ierr);
    ierr = DMSetUp(da); CHKERRQ(ierr);
    ierr = DMSetApplicationContext(da,&user); CHKERRQ(ierr);

    // set field names so that visualization makes sense
    ierr = DMDAGetLocalInfo(da,&info); CHKERRQ(ierr);
    for (k = 0; k < info.dof; k++) {
        ierr = DMDASetFieldName(da,k,(user.field_names)[k]); CHKERRQ(ierr);
    }

    // create TS:  dq/dt = G(t,q)  form
    ierr = TSCreate(PETSC_COMM_WORLD,&ts); CHKERRQ(ierr);
    ierr = TSSetProblemType(ts,TS_NONLINEAR); CHKERRQ(ierr);
    ierr = TSSetDM(ts,da); CHKERRQ(ierr);
    ierr = DMDATSSetRHSFunctionLocal(da,INSERT_VALUES,
           (DMDATSRHSFunctionLocal)FormRHSFunctionLocal,&user); CHKERRQ(ierr);
    ierr = TSSetType(ts,TSRK); CHKERRQ(ierr);  // defaults to -ts_rk_type 3bs

    // set up time axis
    ierr = TSSetTime(ts,user.t0_default); CHKERRQ(ierr);
    ierr = TSSetMaxTime(ts,user.tf_default); CHKERRQ(ierr);
    ierr = TSSetTimeStep(ts,(user.tf_default-user.t0_default)/10.0); CHKERRQ(ierr);
    ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP); CHKERRQ(ierr);
    ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

    // get initial values
    ierr = DMCreateGlobalVector(da,&q); CHKERRQ(ierr);
    ierr = TSGetTime(ts,&t0); CHKERRQ(ierr);
    ierr = FormInitial(&info,q,t0,&user); CHKERRQ(ierr);
    //ierr = VecView(q,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);

    // solve
    hx = (user.b_right - user.a_left) / info.mx;
    ierr = PetscPrintf(PETSC_COMM_WORLD,
               "solving problem %s\n",ProblemTypes[problem]); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,
               "  system of %d equations on %d-point grid with dx=%.5f ...\n",
               info.dof,info.mx,hx); CHKERRQ(ierr);
    ierr = TSSolve(ts,q); CHKERRQ(ierr);

    // report on solution
    ierr = TSGetStepNumber(ts,&steps); CHKERRQ(ierr);
    ierr = TSGetTime(ts,&tf); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,
               "  ... completed %d steps for %.5f <= t <= %.5f\n",
               steps,t0,tf); CHKERRQ(ierr);
    for (k = 0; k < info.dof; k++) {
        ierr = VecStrideNorm(q,k,NORM_INFINITY,&qnorm); CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD,
               "  inf-norm of %s solution: %.5f\n",
               (user.field_names)[k],qnorm); CHKERRQ(ierr);
    }

    VecDestroy(&q);  TSDestroy(&ts);  DMDestroy(&da);
    ierr = DestroyCase(&user); CHKERRQ(ierr);
    return PetscFinalize();
}


PetscErrorCode FormInitial(DMDALocalInfo *info, Vec q, PetscReal t0, ProblemCtx *user) {
    PetscErrorCode   ierr;
    const PetscReal  hx = (user->b_right - user->a_left) / info->mx;
    PetscInt         i;
    PetscReal        x, *aq;

    ierr = DMDAVecGetArray(info->da, q, &aq); CHKERRQ(ierr);
    for (i=info->xs; i<info->xs+info->xm; i++) {
        x = user->a_left + (i+0.5) * hx;
        ierr = user->f_initial(t0,x,&aq[(info->dof)*i]); CHKERRQ(ierr);
    }
    ierr = DMDAVecRestoreArray(info->da, q, &aq); CHKERRQ(ierr);
    return 0;
}


// FIXME implement slope limiters
// FIXME address nonlinear cases of rarefaction and shock

// Right-hand-side of method-of-lines discretization of PDE.  Implements
// Gudonov (Riemann-solver upwind) method.
PetscErrorCode FormRHSFunctionLocal(DMDALocalInfo *info, PetscReal t,
        PetscReal *aq, PetscReal *aG, void *ctx) {
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
        ierr = user->bdryflux_a(t,&aq[(info->dof)*(info->xs)],Fl); CHKERRQ(ierr);
    } else {
        x = user->a_left + (info->xs+0.5) * hx;
        ierr = user->faceflux(t,x-hx/2.0,&aq[(info->dof)*(info->xs-1)],
                                         &aq[(info->dof)*(info->xs)],Fl); CHKERRQ(ierr);
    }

    // i counts cell center
    for (i = info->xs; i < info->xs + info->xm; i++) {
        x = user->a_left + (i+0.5) * hx;
        // aG[n i + k]  filled from g(t,x,u) first
        ierr = user->g_source(t,x,&aq[(info->dof)*i],&aG[(info->dof)*i]); CHKERRQ(ierr);
        // get right-face flux for cell
        if (i == info->mx-1) {
            ierr = user->bdryflux_b(t,&aq[(info->dof)*i],Fr); CHKERRQ(ierr);
        } else {
            ierr = user->faceflux(t,x+hx/2.0,&aq[(info->dof)*i],
                                             &aq[(info->dof)*(i+1)],Fr); CHKERRQ(ierr);
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

