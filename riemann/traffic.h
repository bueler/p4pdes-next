/*
FIXME nonlinear traffic equation  q_t + F(q)_x = 0  where  F(q) = umax * q * (1 - q)
from section 11.1 of LeVeque 2002
*/

// values from:
static const PetscReal traffic_umax = 1.0;

static const char traffic_uname[50] = "q  (traffic density)";

// this initial condition is shown in Figure 11.3 on page 207 of LeVeque
static PetscErrorCode traffic_greenlight(PetscReal t, PetscReal x, PetscReal *q) {
    q[0] = (x < 0.0) ? 1.0 : 0.0;
    return 0;
}

// no source term
static PetscErrorCode traffic_g(PetscReal t, PetscReal x, PetscReal *q, PetscReal *g) {
    g[0] = 0.0;
    return 0;
}

// compute flux at internal faces from left and right values of q = [u]
static PetscErrorCode traffic_faceflux(PetscReal t, PetscReal x,
       PetscReal *ql, PetscReal *qr, PetscReal *F) {
    // see equation (12.4), the Godunov flux function, on page 229 of LeVeque
    // note F(q) = umax q (1 - q), which has peak at qs=1/2
    // we assume umax > 0
    const PetscReal qs = 0.5;   // location of max of F(q)
    PetscReal       Fl, Fr;
    if (ql[0] <= qr[0]) {  // q-values in order
        if (qr[0] <= qs) {                       // both to the left of peak
            F[0] = traffic_umax * ql[0] * (1.0 - ql[0]);
        } else if (ql[0] >= qs) {                // both to the right of peak
            F[0] = traffic_umax * qr[0] * (1.0 - qr[0]);
        } else {                                 // straddling peak
            Fl = traffic_umax * ql[0] * (1.0 - ql[0]);
            Fr = traffic_umax * qr[0] * (1.0 - qr[0]);
            F[0] = PetscMin(Fl,Fr);
        }
    } else {               // q-values out of order
        if (ql[0] <= qs) {                       // both to the left of peak
            F[0] = traffic_umax * ql[0] * (1.0 - ql[0]);
        } else if (qr[0] >= qs) {                // both to the right of peak
            F[0] = traffic_umax * qr[0] * (1.0 - qr[0]);
        } else {                                 // straddling peak
            F[0] = traffic_umax * qs * (1.0 - qs);
        }
    }
    return 0;
}


// FIXME this is only for greenlight
static PetscErrorCode traffic_bdryflux_a(PetscReal t, PetscReal *qr, PetscReal *F) {
    const PetscReal  s = traffic_umax * (1.0 - 2.0 * qr[0]);
    if (s <= 0.0)
        F[0] = traffic_umax * qr[0] * (1.0 - qr[0]);
    else {
        SETERRQ1(PETSC_COMM_SELF,1,
                 "assumption of outflow s <= 0.0 fails at (t=%g,a)\n",t);
    }
    return 0;
}


// FIXME this is only for greenlight
static PetscErrorCode traffic_bdryflux_b(PetscReal t, PetscReal *ql, PetscReal *F) {
    const PetscReal  s = traffic_umax * (1.0 - 2.0 * ql[0]);
    if (s >= 0.0)
        F[0] = traffic_umax * ql[0] * (1.0 - ql[0]);
    else {
        SETERRQ1(PETSC_COMM_SELF,1,
                 "assumption of outflow s >= 0 fails at (t=%g,b)\n",t);
    }
    return 0;
}


// compute maximum speed (of a characteristic) from u; note F'(q) = umax (1 - 2 q)
static PetscErrorCode traffic_maxspeed(PetscReal t, PetscReal x, PetscReal *q,
                                       PetscReal *speed) {
    // see equation (11.16) regarding speed of characteristic
    *speed = PetscAbs(traffic_umax * (1.0 - 2.0 * q[0]));
    return 0;
}

typedef enum {TR_GREENLIGHT} TRInitialType;
static const char* TRInitialTypes[] = {"greenlight",
                                       "TRInitialType", "", NULL};

PetscErrorCode  TrafficInitializer(ProblemCtx *user) {
    PetscErrorCode  ierr;
    TRInitialType   initial = TR_GREENLIGHT;

    if (user == NULL) {
        SETERRQ(PETSC_COMM_SELF,1,"ProblemCtx *user is NULL\n");
    }

    ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"",
               "options for traffic flow solver (-problem traffic)",""); CHKERRQ(ierr);
    ierr = PetscOptionsEnum("-initial", "traffic flow initial condition",
               "riemann.c",TRInitialTypes,(PetscEnum)(initial),(PetscEnum*)&initial,
               NULL); CHKERRQ(ierr);
    ierr = PetscOptionsEnd(); CHKERRQ(ierr);

    user->n_dim = 1;
    ierr = PetscMalloc1(1,&(user->field_names)); CHKERRQ(ierr);
    (user->field_names)[0] = (char*)traffic_uname;
    user->a_left = -30.0; // FIXME from -initial
    user->b_right = 30.0; // FIXME from -initial
    user->t0_default = 0.0;
    user->tf_default = 18.0; // FIXME from -initial
    user->periodic_bcs = PETSC_FALSE;
    user->f_initial = &traffic_greenlight; // FIXME from -initial
    user->g_source = &traffic_g;
    user->bdryflux_a = &traffic_bdryflux_a; // FIXME from -initial
    user->bdryflux_b = &traffic_bdryflux_b; // FIXME from -initial
    user->faceflux = &traffic_faceflux;
    user->maxspeed = &traffic_maxspeed;
    return 0;
}

