static char help[] =
"Solve a hyperbolic system in one space dimension (1D):\n"
"    q_t + F(t,x,q)_x = g(t,x,q)\n"
"where solution q(t,x), flux F(t,x,q), and source g(t,x,q) are column vectors\n"
"of length n.  The domain is (t,x) in [0,T]x[a,b].  The initial condition is\n"
"q(0,x) = f(x).  The flux may be of the form\n"
"    F(t,x,q) = A(t,x,q) q\n"
"but this is not required.  Uses finite volumes (grid values represent\n"
"cell averages) and a case-specific Riemann solver\n"
"    F = faceflux(t,x,qleft,qright)\n"
"at cell faces.  Godunov's method is used: at each cell face use the\n"
"Riemann solver to compute the value on the cell face going forward in time.\n"
"Similarly, case-specific flux boundary conditions are used:\n"
"    F = bdryflux_a(t,qright),\n"
"    F = bdryflux_b(t,qleft),\n"
"Reflecting and outflow boundary conditions are among the implemented types.\n"
"Use option -problem to select the problem case:\n"
"    -problem acoustic   wave equation in system form (n=2) [default]\n"
"    -problem swater     shallow water equations (n=2)\n"
"These PETSc options, among others, give further information and control\n"
"    -da_grid_x M                             [grid of M points]\n"
"    -ts_monitor                              [shows time steps]\n"
"    -ts_monitor_solution draw                [generate simple movie]\n"
"        -draw_pause 0.1 -draw_size 2000,200  [control the movie]\n"
"    -ts_type                                 [default is rk]\n"
"        -ts_rk_type X                        [default is 3bs]\n"
"See the makefile for test examples, and do 'make test' to test.\n\n";


#include <petsc.h>

/* The struct "ProblemCtx" is defined in cases.h.  See comments in cases.h
about how to add new problems. */
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
    ProblemType      problem = ACOUSTIC; // which problem we are solving
    ProblemCtx       user;               // problem-specific information
    PetscInt         k, steps;
    PetscReal        hx, qnorm, t0, tf;

    PetscInitialize(&argc,&argv,(char*)0,help);

    // get which problem we are solving
    // (ProblemType, ProblemTypes, InitializerPtrs are defined in cases.h)
    ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"",
               "riemann (hyperbolic system solver) options",""); CHKERRQ(ierr);
    ierr = PetscOptionsEnum("-problem", "problem type",
               "riemann.c",ProblemTypes,(PetscEnum)(problem),(PetscEnum*)&problem,
               NULL); CHKERRQ(ierr);
    ierr = PetscOptionsEnd(); CHKERRQ(ierr);

    // call the initializer for the given case
    // (it allocates list of strings in user->field_names thus PetscFree below)
    ierr = (*InitializerPtrs[problem])(&user); CHKERRQ(ierr);

    // create grid
    ierr = DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,4,
                        user.n_dim,1,NULL,&da); CHKERRQ(ierr);
    ierr = DMSetFromOptions(da); CHKERRQ(ierr);
    ierr = DMSetUp(da); CHKERRQ(ierr);
    ierr = DMSetApplicationContext(da,&user); CHKERRQ(ierr);
    ierr = DMDAGetLocalInfo(da,&info); CHKERRQ(ierr);
    hx = (user.b_right - user.a_left) / info.mx;
    ierr = DMDASetUniformCoordinates(da,user.a_left+hx/2.0,user.b_right-hx/2.0,
                                     0.0,1.0,0.0,1.0); CHKERRQ(ierr);

    // set field names so that visualization makes sense
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

// FIXME use CFL to get initial dt ... otherwise (at least in swater) one can get negative thickness from a too-large initial dt ... add "maxspeed" function to ProblemCtx?
    ierr = TSSetTimeStep(ts,(user.tf_default-user.t0_default)/10.0); CHKERRQ(ierr);
    ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP); CHKERRQ(ierr);
    ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

    // get initial values
    ierr = DMCreateGlobalVector(da,&q); CHKERRQ(ierr);
    ierr = TSGetTime(ts,&t0); CHKERRQ(ierr);
    ierr = FormInitial(&info,q,t0,&user); CHKERRQ(ierr);
    //ierr = VecView(q,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);

    // solve
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

    // free memory
    VecDestroy(&q);  TSDestroy(&ts);  DMDestroy(&da);
    ierr = PetscFree(user.field_names); CHKERRQ(ierr);
    return PetscFinalize();
}


PetscErrorCode FormInitial(DMDALocalInfo *info, Vec q, PetscReal t0, ProblemCtx *user) {
    PetscErrorCode   ierr;
    const PetscReal  hx = (user->b_right - user->a_left) / info->mx;
    PetscInt         j;
    PetscReal        x, *aq;

    ierr = DMDAVecGetArray(info->da, q, &aq); CHKERRQ(ierr);
    for (j=info->xs; j<info->xs+info->xm; j++) {
        x = user->a_left + (j+0.5) * hx;
        ierr = user->f_initial(t0,x,&aq[(info->dof)*j]); CHKERRQ(ierr);
    }
    ierr = DMDAVecRestoreArray(info->da, q, &aq); CHKERRQ(ierr);
    return 0;
}


// FIXME implement slope limiters

// Right-hand-side of method-of-lines discretization form of PDE.  Implements
// Gudonov (i.e. Riemann-solver upwind) method.
PetscErrorCode FormRHSFunctionLocal(DMDALocalInfo *info, PetscReal t,
        PetscReal *aq, PetscReal *aG, void *ctx) {
    PetscErrorCode   ierr;
    ProblemCtx       *user = (ProblemCtx*)ctx;
    const PetscInt   n = info->dof;
    const PetscReal  hx = (user->b_right - user->a_left) / info->mx;
    PetscInt         j, k;
    PetscReal        x, *Fl, *Fr;     // Fl,Fr hold current cell face fluxes

    //ierr = PetscPrintf(PETSC_COMM_SELF,
    //    "xs=%d,xm=%d,mx=%d,dof=%d\n",info->xs,info->xm,info->mx,info->dof); CHKERRQ(ierr);
    ierr = PetscMalloc1(n,&Fl); CHKERRQ(ierr);
    ierr = PetscMalloc1(n,&Fr); CHKERRQ(ierr);

    // get left-face flux Fl for first cell owned by process; may be at x=a
    if (info->xs == 0) {
        ierr = user->bdryflux_a(t,&aq[n*(info->xs)],Fl); CHKERRQ(ierr);
    } else {
        x = user->a_left + (info->xs+0.5) * hx;
        ierr = user->faceflux(t,x-hx/2.0,&aq[n*(info->xs-1)],
                                         &aq[n*(info->xs)],Fl); CHKERRQ(ierr);
    }

    // for each owned cell, compute RHS  G(t,x,q)
    for (j = info->xs; j < info->xs + info->xm; j++) {   // x_j is cell center
        x = user->a_left + (j+0.5) * hx;
        // set aG[n j + k] = g(t,x_j,u)_k
        ierr = user->g_source(t,x,&aq[n*j],&aG[n*j]); CHKERRQ(ierr);
        // get right-face flux Fr for cell; may be at x=b
        if (j == info->mx-1) {
            ierr = user->bdryflux_b(t,&aq[n*j],Fr); CHKERRQ(ierr);
        } else {
            ierr = user->faceflux(t,x+hx/2.0,&aq[n*j],
                                             &aq[n*(j+1)],Fr); CHKERRQ(ierr);
        }
        // complete the RHS:
        //   aG[n j + k] = g(t,x_j,u)_k + (F_{j-1/2}_k - F_{j+1/2}_k) / hx
        for (k = 0; k < n; k++)
            aG[n*j+k] += (Fl[k] - Fr[k]) / hx;
        // transfer Fr to Fl, for next pass through loop
        for (k = 0; k < info->dof; k++)
            Fl[k] = Fr[k];
    }

    ierr = PetscFree(Fl); CHKERRQ(ierr);
    ierr = PetscFree(Fr); CHKERRQ(ierr);
    return 0;
}

