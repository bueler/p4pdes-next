static char help[] =
"Solve a hyperbolic system in one space dimension (1D):\n"
"    q_t + F(t,x,q)_x = g(t,x,q)\n"
"where solution q(t,x), flux F(t,x,q), and source g(t,x,q) are column vectors\n"
"of length n.  The domain is (t,x) in [0,T] x [a,b].  The initial condition is\n"
"q(0,x) = f(x).  The flux may be of the form\n"
"    F(t,x,q) = A(t,x,q) q\n"
"but this is not required.  Flux boundary conditions are used in most cases,\n"
"    F = bdryflux_a(t,qright)\n"
"    F = bdryflux_b(t,qleft)\n"
"including reflecting and outflow boundary conditions.  Periodic boundary\n"
"conditions are also implemented.\n\n"
"Uses finite volumes---thus grid values represent cell averages---and a\n"
"case-specific Riemann solver\n"
"    F = faceflux(t,x,qleft,qright)\n"
"at cell faces.  Implements the following slope-limiters when computing\n"
"fluxes:\n"
"    -limiter none       Godunov's method, i.e. first-order upwinding\n"
"    -limiter fromm      formula (6.14) in LeVeque 2002\n"
"    -limiter mc         formula (6.29)\n"
"    -limiter minmod     formula (6.26)\n"
"Note that in Godunov's method, at each cell face the Riemann solver computes\n"
"the value of the solution on the cell face going forward in time.\n"
"Control the spatial grid by PETSc option\n"
"    -da_grid_x M                             [grid of M cells/points]\n\n"
"Time stepping is by semi-discretization in space (method of lines) and then\n"
"application of PETSc's (generally) adaptive and higher-order TS solver.\n"
"Control time stepping and solution information by these PETSc options, among\n"
"others:\n"
"    -ts_monitor                              [shows time steps]\n"
"    -ts_monitor_solution draw                [generate simple movie]\n"
"        -draw_pause 0.1 -draw_size 2000,200  [control the movie]\n"
"    -ts_type                                 [default is rk]\n"
"        -ts_rk_type X                        [default is 3bs]\n"
"    -ts_dt 0.01 -ts_adapt_type none          [turn off adaptive]\n\n"
"Use option -problem to select the problem case:\n"
"    -problem acoustic   wave equation in system form (n=2) [default]\n"
"    -problem advection  scalar advection equation (n=1)\n"
"    -problem swater     shallow water equations (n=2)\n"
"    -problem traffic    scalar, nonlinear traffic equation (n=1)\n"
"To see possible initial conditions for problem X do\n"
"    -problem X -help | grep \"initial condition\"\n"
"and then use option\n"
"    -initial Y\n\n"
"See the makefile for test examples, and do 'make test' to test.\n\n";


#include <petsc.h>

/* The struct "ProblemCtx" is defined in cases.h.  See comments in cases.h
about how to add new problems. */
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
    ierr = DMDACreate1d(PETSC_COMM_WORLD,
                        user.periodic_bcs ? DM_BOUNDARY_PERIODIC : DM_BOUNDARY_NONE,
                        4,          // default resolution
                        user.n_dim, // system dimension (d.o.f.)
                        swidth,     // stencil (half) width
                        NULL,&da); CHKERRQ(ierr);
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

    // for each owned cell get slope using the slope-limiter
    ierr = DMCreateLocalVector(info->da,&sig); CHKERRQ(ierr);
    ierr = VecSet(sig,0.0); CHKERRQ(ierr);  // implements limiter == NONE
    ierr = DMDAVecGetArray(info->da,sig,&asig); CHKERRQ(ierr);
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
    ierr = PetscMalloc4(n,&Ql,n,&Qr,n,&Fl,n,&Fr); CHKERRQ(ierr);
    if (info->xs == 0 && user->periodic_bcs == PETSC_FALSE) {
        // use right slope
        slopemodify(n,-hx/2.0,&asig[0],&aq[0],Qr);
        ierr = user->bdryflux_a(t,Qr,Fl); CHKERRQ(ierr);
    } else {
        // use left and right (limited) slope [left is owned by other process]
        x = user->a_left + (info->xs+0.5) * hx;
        slopemodify(n,hx/2.0,&asig[n*(info->xs-1)],&aq[n*(info->xs-1)],Ql);
        slopemodify(n,-hx/2.0,&asig[n*(info->xs)],&aq[n*(info->xs)],Qr);
        ierr = user->faceflux(t,x-hx/2.0,Ql,Qr,Fl); CHKERRQ(ierr);
    }

    // for each owned cell, compute RHS  G(t,x,q)
    for (j = info->xs; j < info->xs + info->xm; j++) {   // x_j is cell center
        x = user->a_left + (j+0.5) * hx;
        // set aG[n j + k] = g(t,x_j,u)_k
        ierr = user->g_source(t,x,&aq[n*j],&aG[n*j]); CHKERRQ(ierr);
        // get right-face flux Fr for cell; may be at x=b
        if (j == info->mx-1 && user->periodic_bcs == PETSC_FALSE) {
            // user left slope
            slopemodify(n,hx/2.0,&asig[n*j],&aq[n*j],Ql);
            ierr = user->bdryflux_b(t,Ql,Fr); CHKERRQ(ierr);
        } else {
            // use left and right (limited) slope; see formulas LeVeque page 193
            slopemodify(n,hx/2.0,&asig[n*j],&aq[n*j],Ql);
            slopemodify(n,-hx/2.0,&asig[n*(j+1)],&aq[n*(j+1)],Qr);
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

