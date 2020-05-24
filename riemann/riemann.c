static char help[] =
"Solve hyperbolic system in 1D:  u_t + (A u)_x = b  where u(t,x) is length n.\n"
"Here A(t,x) is an n x n matrix and b(t,x,u) returns an n-length vector.\n\n";

#include <petsc.h>

extern PetscErrorCode FormInitial(DMDALocalInfo*, Vec, PetscReal);
extern PetscErrorCode FormRHSFunctionLocal(DMDALocalInfo*, PetscReal,
        PetscReal*, PetscReal*, void*);

int main(int argc,char **argv) {
    PetscErrorCode   ierr;
    TS               ts;
    DM               da;
    Vec              u;
    DMDALocalInfo    info;
    PetscReal        hx, t0=0.0, dt=0.1, tf=1.0, L=1.0;
    PetscInt         n=2, steps;

    PetscInitialize(&argc,&argv,(char*)0,help);

    ierr = DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,4,n,1,NULL,&da); CHKERRQ(ierr);
    ierr = DMSetFromOptions(da); CHKERRQ(ierr);
    ierr = DMSetUp(da); CHKERRQ(ierr);
    //ierr = DMSetApplicationContext(da,&user); CHKERRQ(ierr);
    ierr = DMDAGetLocalInfo(da,&info); CHKERRQ(ierr);
    hx = L / info.mx;
    ierr = DMDASetUniformCoordinates(da,    // grid is cell-centered
        0.0+hx/2.0,1.0-hx/2.0,0.0,1.0,0.0,1.0);CHKERRQ(ierr);

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
    ierr = FormInitial(&info,u,t0); CHKERRQ(ierr);

    ierr = PetscPrintf(PETSC_COMM_WORLD,
               "solving on %d grid with dx=%g ...\n",
               info.mx,hx); CHKERRQ(ierr);

    ierr = TSSolve(ts,u); CHKERRQ(ierr);

    ierr = TSGetStepNumber(ts,&steps); CHKERRQ(ierr);
    ierr = TSGetTime(ts,&tf); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,
               "completed %d steps to time %g\n",steps,tf); CHKERRQ(ierr);

    VecDestroy(&u);  TSDestroy(&ts);  DMDestroy(&da);
    return PetscFinalize();
}

// FIXME
PetscErrorCode FormInitial(DMDALocalInfo *info, Vec u, PetscReal t0) {
    PetscErrorCode ierr;
    PetscInt   i, j;
    PetscReal  hx, hy, x, y, **au;

    ierr = VecSet(u,0.0); CHKERRQ(ierr);  // clear it first
    ierr = DMDAVecGetArray(info->da, u, &au); CHKERRQ(ierr);
    hx = 2.0 / info->mx;  hy = 2.0 / info->my;
    for (j=info->ys; j<info->ys+info->ym; j++) {
        y = -1.0 + (j+0.5) * hy;
        for (i=info->xs; i<info->xs+info->xm; i++) {
            x = -1.0 + (i+0.5) * hx;
            switch (user->problem) {
                case STRAIGHT:
                    au[j][i] = (*user->initial_fcn)(x,y);
                    break;
                case ROTATION:
                    au[j][i] = cone(x,y) + box(x,y);
                    break;
                default:
                    SETERRQ(PETSC_COMM_SELF,1,"invalid user->problem\n");
            }
        }
    }
    ierr = DMDAVecRestoreArray(info->da, u, &au); CHKERRQ(ierr);
    return 0;
}

// FIXME
/* method-of-lines discretization gives ODE system  u' = G(t,u)
so our finite volume scheme computes
    G_ij = - (fluxE - fluxW)/hx - (fluxN - fluxS)/hy + g(x,y,U_ij)
but only east (E) and north (N) fluxes are computed
*/
PetscErrorCode FormRHSFunctionLocal(DMDALocalInfo *info, PetscReal t,
        PetscReal **au, PetscReal **aG, AdvectCtx *user) {
    PetscInt   i, j, q, dj, di;
    PetscReal  hx, hy, halfx, halfy, x, y, a,
               u_up, u_dn, u_far, theta, flux;

    // clear G first
    for (j = info->ys; j < info->ys + info->ym; j++)
        for (i = info->xs; i < info->xs + info->xm; i++)
            aG[j][i] = 0.0;
    // fluxes on cell boundaries are traversed in E,N order with indices
    // q=0 for E and q=1 for N; cell center has coordinates (x,y)
    hx = 2.0 / info->mx;  hy = 2.0 / info->my;
    halfx = hx / 2.0;     halfy = hy / 2.0;
    for (j = info->ys-1; j < info->ys + info->ym; j++) { // note -1 start
        y = -1.0 + (j+0.5) * hy;
        for (i = info->xs-1; i < info->xs + info->xm; i++) { // -1 start
            x = -1.0 + (i+0.5) * hx;
            if ((i >= info->xs) && (j >= info->ys)) {
                aG[j][i] += g_source(x,y,au[j][i],user);
            }
            for (q = 0; q < 2; q++) {   // E (q=0) and N (q=1) bdry fluxes
                if (q == 0 && j < info->ys)  continue;
                if (q == 1 && i < info->xs)  continue;
                di = 1 - q;
                dj = q;
                a = a_wind(x + halfx*di,y + halfy*dj,q,user);
                // first-order flux
                u_up = (a >= 0.0) ? au[j][i] : au[j+dj][i+di];
                flux = a * u_up;
                // use flux-limiter
                if (user->limiter_fcn != NULL) {
                    // formulas (1.2),(1.3),(1.6); H&V pp 216--217
                    u_dn = (a >= 0.0) ? au[j+dj][i+di] : au[j][i];
                    if (u_dn != u_up) {
                        u_far = (a >= 0.0) ? au[j-dj][i-di]
                                           : au[j+2*dj][i+2*di];
                        theta = (u_up - u_far) / (u_dn - u_up);
                        flux += a * (*user->limiter_fcn)(theta)
                                  * (u_dn-u_up);
                    }
                }
                // update owned G_ij on both sides of computed flux
                if (q == 0) {
                    if (i >= info->xs)
                        aG[j][i]   -= flux / hx;
                    if (i+1 < info->xs + info->xm)
                        aG[j][i+1] += flux / hx;
                } else {
                    if (j >= info->ys)
                        aG[j][i]   -= flux / hy;
                    if (j+1 < info->ys + info->ym)
                        aG[j+1][i] += flux / hy;
                }
            }
        }
    }
    return 0;
}

