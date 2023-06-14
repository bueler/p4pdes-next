static char help[] =
"Solve a hyperbolic system in one space dimension (1D):\n"
"    q_t + F(t,x,q)_x = g(t,x,q)\n"
"where solution q(t,x), flux F(t,x,q), and source g(t,x,q) are column vectors\n"
"of length n.  The domain is (t,x) in [0,T] x [a,b].  The initial condition is\n"
"q(0,x) = f(x).  The flux may be of the form\n"
"    F(t,x,q) = A(t,x,q) q\n"
"but this is not required.  Flux boundary conditions are used in all cases,\n"
"    F = bdryflux_a(t,qright)\n"
"    F = bdryflux_b(t,qleft)\n"
"including nonreflecting, reflecting, outflow, and periodic boundary\n"
"conditions.\n\n"
"Uses a finite volume discretization so the grid values represent cell\n"
"averages.  Each case implements a Riemann solver\n"
"    F = faceflux(t,x,qleft,qright)\n"
"at cell faces.  (The Riemann solver computes the value of the solution on\n"
"the cell face going forward in time.)  Implements the following\n"
"slope-limiters when computing fluxes:\n"
"    -limiter none       Godunov's method, i.e. first-order upwinding\n"
"    -limiter fromm      formula (6.14) in LeVeque 2002\n"
"    -limiter mc         formula (6.29)\n"
"    -limiter minmod     formula (6.26)\n"
"Control the spatial grid by PETSc option\n"
"    -da_grid_x M                             [grid of M cells/points]\n\n"
"Time stepping is by semi-discretization in space (method of lines) and then\n"
"application of PETSc's (generally) adaptive and higher-order TS solvers.\n"
"Control time stepping and solution information by these PETSc options, among\n"
"others:\n"
"    -ts_monitor                              [shows time steps]\n"
"    -ts_monitor_solution draw                [generate simple movie]\n"
"        -draw_pause 0.1 -draw_size 2000,200  [control the movie]\n"
"    -ts_type                                 [default is rk]\n"
"        -ts_rk_type X                        [default is 3bs]\n"
"    -ts_dt 0.01 -ts_adapt_type none          [turn off adaptive]\n"
"Note that TS solver type SSP (-ts_type ssp) is recommended for these\n"
"hyperbolic problems.\n\n"
"Use option -problem selects the problem case:\n"
"    -problem acoustic   wave equation in system form (n=2) [default]\n"
"    -problem advection  scalar advection equation (n=1)\n"
"    -problem swater     shallow water equations (n=2)\n"
"    -problem traffic    scalar, nonlinear traffic equation (n=1)\n"
"To see possible initial conditions for problem X see the corresponding\n"
".h file.  Based on the problem-specific possiblities, set\n"
"    -initial Y\n\n"
"See the makefile for test examples, and do 'make test' to test.\n\n"
"This program is documented by the slides in the fvolume/ directory at\n"
"https://github.com/bueler/slide-teach\n\n";


#include <petsc.h>

/* The struct "ProblemCtx" is defined in cases.h.  The comments in cases.h
show how to add new problems. */
#include "cases.h"

// minmod(a,b) as define on LeVeque page 111
static PetscReal minmod(PetscReal a, PetscReal b) {
    if (a*b > 0) // both signs agree
        return (PetscAbs(a) < PetscAbs(b)) ? a : b;
    else
        return 0.0;
}

typedef enum {NONE,FROMM,MC,MINMOD} LimiterType;
static const char* LimiterTypes[] = {"none","fromm","mc","minmod",
                                     "LimiterType", "", NULL};

static LimiterType limiter = NONE;     // slope-limiter

extern PetscErrorCode FormInitial(DMDALocalInfo*, Vec, PetscReal, ProblemCtx*);
extern PetscErrorCode GetMaxSpeed(DMDALocalInfo*, Vec, PetscReal, PetscReal*, ProblemCtx*);
extern PetscErrorCode FormRHSFunctionLocal(DMDALocalInfo*, PetscReal,
        PetscReal*, PetscReal*, void*);

int main(int argc,char **argv) {
    TS               ts;                 // ODE solver for method-of-lines (MOL)
    DM               da;                 // structured grid
    Vec              q;                  // the solution
    DMDALocalInfo    info;               // structured grid info
    ProblemType      problem = ACOUSTIC; // which problem we are solving
    ProblemCtx       user;               // problem-specific information
    PetscInt         swidth, k, steps;
    PetscBool        flg;
    PetscReal        hx, qmin, qmax, t0, tf, dt, c;

    PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));

    // get which problem we are solving
    // (ProblemType, ProblemTypes, InitializerPtrs are defined in cases.h)
    PetscOptionsBegin(PETSC_COMM_WORLD,"",
               "riemann (hyperbolic system solver) options","");
    PetscCall(PetscOptionsEnum("-problem", "problem type",
               "riemann.c",ProblemTypes,(PetscEnum)(problem),(PetscEnum*)&problem,
               NULL));
    PetscCall(PetscOptionsEnum("-limiter", "limiter type",
               "riemann.c",LimiterTypes,(PetscEnum)(limiter),(PetscEnum*)&limiter,
               NULL));
    PetscOptionsEnd();

    // call the initializer for the given case
    // (it allocates list of strings in user->field_names thus PetscFree below)
    PetscCall((*InitializerPtrs[problem])(&user));

    // create grid
    swidth = (limiter == NONE) ? 1 : 2;
    PetscCall(DMDACreate1d(PETSC_COMM_WORLD,
                        user.periodic_bcs ? DM_BOUNDARY_PERIODIC : DM_BOUNDARY_NONE,
                        4,          // default resolution
                        user.n_dim, // system dimension (d.o.f.)
                        swidth,     // stencil (half) width
                        NULL,&da));
    PetscCall(DMSetFromOptions(da));
    PetscCall(DMSetUp(da));
    PetscCall(DMSetApplicationContext(da,&user));
    PetscCall(DMDAGetLocalInfo(da,&info));
    hx = (user.b_right - user.a_left) / info.mx;
    PetscCall(DMDASetUniformCoordinates(da,user.a_left+hx/2.0,user.b_right-hx/2.0,
                                     0.0,1.0,0.0,1.0));

    // set field names so that visualization makes sense
    for (k = 0; k < info.dof; k++) {
        PetscCall(DMDASetFieldName(da,k,(user.field_names)[k]));
    }

    // create TS:  dq/dt = G(t,q)  form
    PetscCall(TSCreate(PETSC_COMM_WORLD,&ts));
    PetscCall(TSSetProblemType(ts,TS_NONLINEAR));
    PetscCall(TSSetDM(ts,da));
    PetscCall(DMDATSSetRHSFunctionLocal(da,INSERT_VALUES,
           (DMDATSRHSFunctionLocal)FormRHSFunctionLocal,&user));
    PetscCall(TSSetType(ts,TSRK));  // defaults to -ts_rk_type 3bs

    // set up time axis
    PetscCall(TSSetTime(ts,user.t0_default));
    PetscCall(TSSetMaxTime(ts,user.tf_default));
    dt = user.tf_default - user.t0_default;
    PetscCall(TSSetTimeStep(ts,dt));  // usually reset below
    PetscCall(TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP));
    PetscCall(TSSetFromOptions(ts));

    // get initial values
    PetscCall(DMCreateGlobalVector(da,&q));
    PetscCall(TSGetTime(ts,&t0));
    PetscCall(FormInitial(&info,q,t0,&user));
    //PetscCall(VecView(q,PETSC_VIEWER_STDOUT_WORLD));

    // use CFL to reset initial time-step dt (unless user sets)
    PetscCall(PetscOptionsHasName(NULL,NULL,"-ts_dt",&flg));
    PetscCall(GetMaxSpeed(&info,q,t0,&c,&user));
    if (!flg && c > 0.0) {
        PetscCall(TSGetMaxTime(ts,&tf));
        dt = PetscMin(hx / c, tf-t0);
        PetscCall(TSSetTimeStep(ts,dt));
    } else {
        PetscCall(TSGetTimeStep(ts,&dt));
    }

    // solve
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
               "solving problem %s, a system of %d equations,\n"
               "  on %d-point grid with dx=%.6f and initial dt=%.6f...\n",
               ProblemTypes[problem],info.dof,info.mx,hx,dt));
    PetscCall(TSSolve(ts,q));

    // report on solution
    PetscCall(TSGetStepNumber(ts,&steps));
    PetscCall(TSGetTime(ts,&tf));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
               "  ... completed %d steps for %.4f <= t <= %.4f\n",
               steps,t0,tf));
    for (k = 0; k < info.dof; k++) {
        PetscCall(VecStrideMin(q,k,NULL,&qmin));
        PetscCall(VecStrideMax(q,k,NULL,&qmax));
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,
               "  range of %d component: %8.5f <= %s <= %8.5f\n",
               k,qmin,(user.field_names)[k],qmax));
    }

    // free memory
    VecDestroy(&q);  TSDestroy(&ts);  DMDestroy(&da);
    PetscCall(PetscFree(user.field_names));
    PetscCall(PetscFinalize());
    return 0;
}


PetscErrorCode FormInitial(DMDALocalInfo *info, Vec q, PetscReal t0, ProblemCtx *user) {
    const PetscReal  hx = (user->b_right - user->a_left) / info->mx;
    PetscInt         j;
    PetscReal        x, *aq;

    PetscCall(DMDAVecGetArray(info->da, q, &aq));
    for (j=info->xs; j<info->xs+info->xm; j++) {
        x = user->a_left + (j+0.5) * hx;
        PetscCall(user->f_initial(t0,x,&aq[(info->dof)*j]));
    }
    PetscCall(DMDAVecRestoreArray(info->da, q, &aq));
    return 0;
}


PetscErrorCode GetMaxSpeed(DMDALocalInfo *info, Vec q, PetscReal t,
                           PetscReal *maxspeed, ProblemCtx *user) {
    const PetscReal  hx = (user->b_right - user->a_left) / info->mx;
    PetscInt         j;
    PetscReal        x, cj, locmax, *aq;
    MPI_Comm         comm;

    PetscCall(DMDAVecGetArray(info->da, q, &aq));
    locmax = 0.0;
    for (j=info->xs; j<info->xs+info->xm; j++) {
        x = user->a_left + (j+0.5) * hx;
        PetscCall(user->maxspeed(t,x,&aq[(info->dof)*j],&cj));
        locmax = PetscMax(locmax,cj);
    }
    PetscCall(DMDAVecRestoreArray(info->da, q, &aq));
    PetscCall(PetscObjectGetComm((PetscObject)info->da,&comm));
    PetscCall(MPI_Allreduce(&locmax,maxspeed,1,MPIU_REAL,MPIU_MAX,comm));
    return 0;
}


static inline void ncopy(PetscInt n, PetscReal *src, PetscReal *tgt) {
    PetscInt k;
    for (k = 0; k < n; k++)
        tgt[k] = src[k];
}

static inline void slopemodify(PetscInt n, PetscReal C,
                               PetscReal *sig, PetscReal *Q_in,
                               PetscReal *Q_out) {
    PetscInt k;
    for (k = 0; k < n; k++)
        Q_out[k] = Q_in[k] + C * sig[k];
}

// Right-hand-side of method-of-lines discretization form of PDE.  Implements
// Gudonov (i.e. Riemann-solver upwind) method with a slope limiter.
PetscErrorCode FormRHSFunctionLocal(DMDALocalInfo *info, PetscReal t,
        PetscReal *aq, PetscReal *aG, void *ctx) {
    ProblemCtx       *user = (ProblemCtx*)ctx;
    const PetscInt   n = info->dof;
    const PetscReal  hx = (user->b_right - user->a_left) / info->mx;
    Vec              sig;
    PetscInt         j, k;
    PetscReal        x, *asig, sl, sr,
                     *Ql, *Qr,  // slope-limited values of solution at either
                                //     side of current face
                     *Fl, *Fr;  // face fluxes on either end of current cell

    // for each owned cell get slope using the slope-limiter
    PetscCall(DMCreateLocalVector(info->da,&sig));
    PetscCall(VecSet(sig,0.0));  // implements limiter == NONE
    PetscCall(DMDAVecGetArray(info->da,sig,&asig));
    if (limiter != NONE) {  // following block assumes swidth >= 2
        for (j = info->xs-1; j < info->xs + info->xm+1; j++) {   // x_j is cell center
            for (k = 0; k < n; k++) {
                if ((j < 0 || j > info->mx-1) && user->periodic_bcs == PETSC_FALSE)
                    continue;
                if (j == 0 && user->periodic_bcs == PETSC_FALSE)
                    asig[n*j+k] = (aq[n*(j+1) + k] - aq[n*j + k]) / hx;
                else if (j == info->mx-1 && user->periodic_bcs == PETSC_FALSE)
                    asig[n*j+k] = (aq[n*j + k] - aq[n*(j-1) + k]) / hx;
                else {
                    if (limiter == FROMM) {
                        asig[n*j+k] = (aq[n*(j+1) + k] - aq[n*(j-1) + k]) / (2.0 * hx);
                    } else {
                        sr = (aq[n*(j+1) + k] - aq[n*j + k]) / hx;
                        sl = (aq[n*j + k] - aq[n*(j-1) + k]) / hx;
                        if (limiter == MINMOD)
                            asig[n*j+k] = minmod(sl,sr);
                        else if (limiter == MC)
                            asig[n*j+k] = minmod(0.5*(sl+sr),2.0*minmod(sl,sr));
                        else {
                            SETERRQ(PETSC_COMM_SELF,1,"how did I get here?\n");
                        }
                    }
                }
            }
        }
    }

    // get left-face flux Fl for first cell owned by process; may be at x=a
    PetscCall(PetscMalloc4(n,&Ql,n,&Qr,n,&Fl,n,&Fr));
    if (info->xs == 0 && user->periodic_bcs == PETSC_FALSE) {
        // use right slope
        slopemodify(n,-hx/2.0,&asig[0],&aq[0],Qr);
        PetscCall(user->bdryflux_a(t,hx,Qr,Fl));
    } else {
        // use left and right (limited) slope [left is owned by other process]
        x = user->a_left + (info->xs+0.5) * hx;
        slopemodify(n,hx/2.0,&asig[n*(info->xs-1)],&aq[n*(info->xs-1)],Ql);
        slopemodify(n,-hx/2.0,&asig[n*(info->xs)],&aq[n*(info->xs)],Qr);
        PetscCall(user->faceflux(t,x-hx/2.0,Ql,Qr,Fl));
    }

    // for each owned cell, compute RHS  G(t,x,q)
    for (j = info->xs; j < info->xs + info->xm; j++) {   // x_j is cell center
        x = user->a_left + (j+0.5) * hx;
        // set aG[n j + k] = g(t,x_j,u)_k
        PetscCall(user->g_source(t,x,&aq[n*j],&aG[n*j]));
        // get right-face flux Fr for cell; may be at x=b
        if (j == info->mx-1 && user->periodic_bcs == PETSC_FALSE) {
            // user left slope
            slopemodify(n,hx/2.0,&asig[n*j],&aq[n*j],Ql);
            PetscCall(user->bdryflux_b(t,hx,Ql,Fr));
        } else {
            // use left and right (limited) slope; see formulas LeVeque page 193
            slopemodify(n,hx/2.0,&asig[n*j],&aq[n*j],Ql);
            slopemodify(n,-hx/2.0,&asig[n*(j+1)],&aq[n*(j+1)],Qr);
            PetscCall(user->faceflux(t,x+hx/2.0,Ql,Qr,Fr));
        }
        // complete the RHS:
        //   aG[n j + k] = g(t,x_j,u)_k + (F_{j-1/2}_k - F_{j+1/2}_k) / hx
        for (k = 0; k < n; k++)
            aG[n*j+k] += (Fl[k] - Fr[k]) / hx;
        ncopy(n,Fr,Fl); // transfer Fr to Fl, for next loop
    }

    // clean up
    PetscCall(PetscFree4(Ql,Qr,Fl,Fr));
    PetscCall(DMDAVecRestoreArray(info->da,sig,&asig));
    PetscCall(VecDestroy(&sig));
    return 0;
}

