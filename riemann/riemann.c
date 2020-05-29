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
"    -ts_dt 0.01 -ts_adapt_type none          [turn off adaptive]\n"
"See the makefile for test examples, and do 'make test' to test.\n\n";


#include <petsc.h>

/* The struct "ProblemCtx" is defined in cases.h.  See comments in cases.h
about how to add new problems. */
#include "cases.h"

// minmod(a,b) as define on LeVeque page 111
static PetscReal minmod(PetscReal a, PetscReal b) {
    if (a*b > 0)
        return (PetscAbs(a) < PetscAbs(b)) ? a : b;
    else
        return 0.0;
}

// FIXME superbee

// FIXME mc

typedef enum {NONE,
              MINMOD} LimiterType;
static const char* LimiterTypes[] = {"none",
                                     "minmod",
                                     "LimiterType", "", NULL};

static LimiterType limiter = NONE;     // slope-limiter

extern PetscErrorCode FormInitial(DMDALocalInfo*, Vec, PetscReal, ProblemCtx*);
extern PetscErrorCode GetMaxSpeed(DMDALocalInfo*, Vec, PetscReal, PetscReal*, ProblemCtx*);
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
    PetscInt         swidth, k, steps;
    PetscBool        flg;
    PetscReal        hx, qnorm, t0, tf, dt, c;

    PetscInitialize(&argc,&argv,(char*)0,help);

    // get which problem we are solving
    // (ProblemType, ProblemTypes, InitializerPtrs are defined in cases.h)
    ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"",
               "riemann (hyperbolic system solver) options",""); CHKERRQ(ierr);
    ierr = PetscOptionsEnum("-problem", "problem type",
               "riemann.c",ProblemTypes,(PetscEnum)(problem),(PetscEnum*)&problem,
               NULL); CHKERRQ(ierr);
    ierr = PetscOptionsEnum("-limiter", "limiter type",
               "riemann.c",LimiterTypes,(PetscEnum)(limiter),(PetscEnum*)&limiter,
               NULL); CHKERRQ(ierr);
    ierr = PetscOptionsEnd(); CHKERRQ(ierr);

    // call the initializer for the given case
    // (it allocates list of strings in user->field_names thus PetscFree below)
    ierr = (*InitializerPtrs[problem])(&user); CHKERRQ(ierr);

    // create grid
    swidth = (limiter == NONE) ? 1 : 2;
    ierr = DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,4,
                        user.n_dim,swidth,NULL,&da); CHKERRQ(ierr);
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
    dt = user.tf_default - user.t0_default;
    ierr = TSSetTimeStep(ts,dt); CHKERRQ(ierr);  // usually reset below
    ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP); CHKERRQ(ierr);
    ierr = TSSetFromOptions(ts); CHKERRQ(ierr);

    // get initial values
    ierr = DMCreateGlobalVector(da,&q); CHKERRQ(ierr);
    ierr = TSGetTime(ts,&t0); CHKERRQ(ierr);
    ierr = FormInitial(&info,q,t0,&user); CHKERRQ(ierr);
    //ierr = VecView(q,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);

    // use CFL to reset initial time-step dt (unless user sets)
    ierr = PetscOptionsHasName(NULL,NULL,"-ts_dt",&flg); CHKERRQ(ierr);
    ierr = GetMaxSpeed(&info,q,t0,&c,&user); CHKERRQ(ierr);
    if (!flg && c > 0.0) {
        ierr = TSGetMaxTime(ts,&tf); CHKERRQ(ierr);
        dt = PetscMin(hx / c, tf-t0);
        ierr = TSSetTimeStep(ts,dt); CHKERRQ(ierr);
    } else {
        ierr = TSGetTimeStep(ts,&dt); CHKERRQ(ierr);
    }

    // solve
    ierr = PetscPrintf(PETSC_COMM_WORLD,
               "solving problem %s, a system of %d equations,\n"
               "  on %d-point grid with dx=%.6f and initial dt=%.6f...\n",
               ProblemTypes[problem],info.dof,info.mx,hx,dt); CHKERRQ(ierr);
    ierr = TSSolve(ts,q); CHKERRQ(ierr);

    // report on solution
    ierr = TSGetStepNumber(ts,&steps); CHKERRQ(ierr);
    ierr = TSGetTime(ts,&tf); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,
               "  ... completed %d steps for %.4f <= t <= %.4f\n",
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


PetscErrorCode GetMaxSpeed(DMDALocalInfo *info, Vec q, PetscReal t,
                           PetscReal *maxspeed, ProblemCtx *user) {
    PetscErrorCode   ierr;
    const PetscReal  hx = (user->b_right - user->a_left) / info->mx;
    PetscInt         j;
    PetscReal        x, cj, locmax, *aq;
    MPI_Comm         comm;

    ierr = DMDAVecGetArray(info->da, q, &aq); CHKERRQ(ierr);
    locmax = 0.0;
    for (j=info->xs; j<info->xs+info->xm; j++) {
        x = user->a_left + (j+0.5) * hx;
        ierr = user->maxspeed(t,x,&aq[(info->dof)*j],&cj); CHKERRQ(ierr);
        locmax = PetscMax(locmax,cj);
    }
    ierr = DMDAVecRestoreArray(info->da, q, &aq); CHKERRQ(ierr);
    ierr = PetscObjectGetComm((PetscObject)info->da,&comm); CHKERRQ(ierr);
    ierr = MPI_Allreduce(&locmax,maxspeed,1,MPIU_REAL,MPIU_MAX,comm); CHKERRQ(ierr);
    return 0;
}


static inline void ncopy(PetscInt n, PetscReal *src, PetscReal *tgt) {
    PetscInt k;
    for (k = 0; k < n; k++)
        tgt[k] = src[k];
}

static inline void slopemodify(PetscInt n, PetscReal hx,
                               PetscReal *sigl, PetscReal *sigr,
                               PetscReal *Ql, PetscReal *Qr) {
    PetscInt k;
    // formulas on LeVeque page 193
    for (k = 0; k < n; k++) {
        Ql[k] += (hx/2.0) * sigl[k];
        Qr[k] -= (hx/2.0) * sigr[k];
    }
}

// Right-hand-side of method-of-lines discretization form of PDE.  Implements
// Gudonov (i.e. Riemann-solver upwind) method with a slope limiter.
PetscErrorCode FormRHSFunctionLocal(DMDALocalInfo *info, PetscReal t,
        PetscReal *aq, PetscReal *aG, void *ctx) {
    PetscErrorCode   ierr;
    ProblemCtx       *user = (ProblemCtx*)ctx;
    const PetscInt   n = info->dof;
    const PetscReal  hx = (user->b_right - user->a_left) / info->mx;
    Vec              sig;
    PetscInt         j, k;
    PetscReal        x, *asig, sl, sr,
                     *Ql, *Qr,  // slope-limited values of solution at either
                                //     side of current face
                     *Fl, *Fr;  // face fluxes on either end of current cell

    // for each owned cell get limited slope
    ierr = DMCreateLocalVector(info->da,&sig); CHKERRQ(ierr);
    ierr = VecSet(sig,0.0); CHKERRQ(ierr);  // implements limiter == NONE
    ierr = DMDAVecGetArray(info->da,sig,&asig); CHKERRQ(ierr);
    if (limiter == MINMOD) {
        for (j = info->xs-1; j < info->xs + info->xm+1; j++) {   // x_j is cell center
            for (k = 0; k < n; k++) {
                if (j < 0 || j > info->mx-1)
                    continue;
                if (j == 0 || j == info->mx-1) {
                    asig[n*j+k] = 0.0;  // FIXME compute slope?
                } else {
                    sl = (aq[n*j + k] - aq[n*(j-1) + k]) / hx;
                    sr = (aq[n*(j+1) + k] - aq[n*j + k]) / hx;
                    asig[n*j+k] = minmod(sl,sr);
                }
            }
        }
    }

    // get left-face flux Fl for first cell owned by process; may be at x=a
    ierr = PetscMalloc4(n,&Ql,n,&Qr,n,&Fl,n,&Fr); CHKERRQ(ierr);
    if (info->xs == 0) {
        // FIXME slope limit right?
        ierr = user->bdryflux_a(t,&aq[n*(info->xs)],Fl); CHKERRQ(ierr);
    } else {
        x = user->a_left + (info->xs+0.5) * hx;
        ncopy(n,&aq[n*(info->xs-1)],Ql);
        ncopy(n,&aq[n*(info->xs)],Qr);
        slopemodify(n,hx,&asig[n*(info->xs-1)],&asig[n*(info->xs)],Ql,Qr);
        ierr = user->faceflux(t,x-hx/2.0,Ql,Qr,Fl); CHKERRQ(ierr);
    }

    // for each owned cell, compute RHS  G(t,x,q)
    for (j = info->xs; j < info->xs + info->xm; j++) {   // x_j is cell center
        x = user->a_left + (j+0.5) * hx;
        // set aG[n j + k] = g(t,x_j,u)_k
        ierr = user->g_source(t,x,&aq[n*j],&aG[n*j]); CHKERRQ(ierr);
        // get right-face flux Fr for cell; may be at x=b
        if (j == info->mx-1) {
            // FIXME slope limit left?
            ierr = user->bdryflux_b(t,&aq[n*j],Fr); CHKERRQ(ierr);
        } else {
            ncopy(n,&aq[n*j],Ql);
            ncopy(n,&aq[n*(j+1)],Qr);
            slopemodify(n,hx,&asig[n*j],&asig[n*(j+1)],Ql,Qr);
            ierr = user->faceflux(t,x+hx/2.0,Ql,Qr,Fr); CHKERRQ(ierr);
        }
        // complete the RHS:
        //   aG[n j + k] = g(t,x_j,u)_k + (F_{j-1/2}_k - F_{j+1/2}_k) / hx
        for (k = 0; k < n; k++)
            aG[n*j+k] += (Fl[k] - Fr[k]) / hx;
        ncopy(n,Fr,Fl); // transfer Fr to Fl, for next loop
    }

    // clean up
    ierr = PetscFree4(Ql,Qr,Fl,Fr); CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(info->da,sig,&asig); CHKERRQ(ierr);
    ierr = VecDestroy(&sig); CHKERRQ(ierr);
    return 0;
}

