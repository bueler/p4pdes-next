/* FIXME  document eigenvectors, Riemann invariants, local face flux computation from Riemann solver

from LeVeque [1] pages 254--257; specific problem is for Figure 13.1

uses Roe solver designed for shallow water, section 15.3.3

for constants see setprob.data and qinit.f in download from
http://depts.washington.edu/clawpack/clawpack-4.3/book/chap13/swhump1/www/index.html
thus:
grav = 1.0
beta = 5.0
q0(x) = 1.0 + 0.4 exp(-beta x^2)
q1(x) = 0.0

[1] R. LeVeque, "Finite Volume Methods for Hyperbolic Problems", Cambridge
  University Press, 2002.
*/

const PetscReal swater_grav = 1.0;

const char swater_hname[50] = "h (surface height)",
           swater_huname[50] = "h u (height * velocity)";

// see LeVeque Figure 13.1
PetscErrorCode swater_hump(PetscReal t, PetscReal x, PetscReal *q) {
    q[0] = 1.0 + 0.4 * PetscExpReal(- 5.0 * x*x);
    q[1] = 0.0;
    return 0;
}

// see LeVeque Figure 13.4
PetscErrorCode swater_dam(PetscReal t, PetscReal x, PetscReal *q) {
    q[0] = (x < 0) ? 3.0 : 1.0;
    q[1] = 0.0;
    return 0;
}

PetscErrorCode swater_g(PetscReal t, PetscReal x, PetscReal *q, PetscReal *g) {
    g[0] = 0.0;
    g[1] = 0.0;
    return 0;
}

// FIXME for quick-and-dirty outflow
extern PetscErrorCode swater_faceflux(PetscReal,PetscReal,PetscReal*,PetscReal*,PetscReal*);

// FIXME want outflow boundary condition at x=a
PetscErrorCode swater_bdryflux_a(PetscReal t, PetscReal *qr, PetscReal *F) {
    PetscErrorCode ierr;
    if (qr[0] <= 0.0) {
        SETERRQ2(PETSC_COMM_SELF,1,
                 "h = qr[0] = %g is nonpositive at (t=%g,a)\n",
                 qr[0],t);
    }
    ierr = swater_faceflux(t,-5.0,qr,qr,F); CHKERRQ(ierr);
    return 0;
}

// FIXME want outflow boundary condition at x=b
PetscErrorCode swater_bdryflux_b(PetscReal t, PetscReal *ql, PetscReal *F) {
    PetscErrorCode ierr;
    if (ql[0] <= 0.0) {
        SETERRQ2(PETSC_COMM_SELF,1,
                 "h = qr[0] = %g is nonpositive at (t=%g,b)\n",
                 ql[0],t);
    }
    ierr = swater_faceflux(t,5.0,ql,ql,F); CHKERRQ(ierr);
    return 0;
}

// compute flux at internal faces from left and right values of q = [h, h u]
PetscErrorCode swater_faceflux(PetscReal t, PetscReal x,
       PetscReal *ql, PetscReal *qr, PetscReal *F) {
    // assert h > 0 on both sides
    if (ql[0] <= 0.0) {
        SETERRQ3(PETSC_COMM_SELF,1,
                 "h = ql[0] = %g is nonpositive at (t,x) = (%g,%g)\n",
                 ql[0],t,x);
    }
    if (qr[0] <= 0.0) {
        SETERRQ3(PETSC_COMM_SELF,2,
                 "h = qr[0] = %g is nonpositive at (t,x) = (%g,%g)\n",
                 qr[0],t,x);
    }
    // compute hbar and uhat from Roe averages, section 15.3.3 of LeVeque
    const PetscReal hbar = 0.5 * (ql[0] + qr[0]),
                    rhl  = PetscSqrtReal(ql[0]),
                    rhr  = PetscSqrtReal(qr[0]),
                    ul   = ql[1] / ql[0],
                    ur   = qr[1] / qr[0],
                    uhat = (rhl * ul + rhr * ur) / (rhl + rhr),
                    delta = PetscSqrtReal(swater_grav * hbar),
                    lam0 = uhat - delta,   // note lam0 < lam1 always
                    lam1 = uhat + delta;
    // Riemann solver using lam0, lam1 speeds
    PetscReal hface, huface;
    if (lam1 < 0.0) {          // both speeds negative
        hface = qr[0];
        huface = qr[1];
    } else if (lam0 >= 0.0) {  // both speeds nonnegative
        hface = ql[0];
        huface = ql[1];
    } else {
        const PetscReal beta0 = - uhat - delta,
                        beta1 = - uhat + delta,
                        v0r   = beta0 * qr[0] + qr[1],
                        v1l   = beta1 * ql[0] + ql[1];
        hface = (v1l - v0r) / (beta1 - beta0);
        huface = (beta1 * v0r - beta0 * v1l) / (beta1 - beta0);
    }
    // for flux formula see page 255 of LeVeque
    F[0] = huface;
    F[1] = (huface*huface / hface) + 0.5 * swater_grav * hface*hface;
    return 0;
}


typedef enum {HUMP,
              DAM} SWInitialType;
static const char* SWInitialTypes[] = {"hump",
                                       "dam",
                                       "SWInitialType", "", NULL};

PetscErrorCode  SWaterInitializer(ProblemCtx *user) {
    PetscErrorCode  ierr;
    SWInitialType   initial = HUMP;

    if (user == NULL) {
        SETERRQ(PETSC_COMM_SELF,1,"ProblemCtx *user is NULL\n");
    }

    ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"",
               "options for shallow water solver (-problem swater)",""); CHKERRQ(ierr);
    ierr = PetscOptionsEnum("-initial", "shallow water initial condition",
               "riemann.c",SWInitialTypes,(PetscEnum)(initial),(PetscEnum*)&initial,
               NULL); CHKERRQ(ierr);
    ierr = PetscOptionsEnd(); CHKERRQ(ierr);
    user->n_dim = 2;
    ierr = PetscMalloc1(2,&(user->field_names)); CHKERRQ(ierr);
    (user->field_names)[0] = (char*)swater_hname;
    (user->field_names)[1] = (char*)swater_huname;
    user->a_left = -5.0;
    user->b_right = 5.0;
    user->t0_default = 0.0;
    user->tf_default = 3.0;
    user->f_initial = (initial == HUMP) ? &swater_hump : &swater_dam;
    user->g_source = &swater_g;
    user->bdryflux_a = &swater_bdryflux_a;
    user->bdryflux_b = &swater_bdryflux_b;
    user->faceflux = &swater_faceflux;
    return 0;
}

