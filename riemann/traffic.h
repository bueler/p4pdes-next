/*
This scalar, nonlinear traffic problem is stated in equations (11.1) and (11.6)
in section 11.1 of LeVeque [1]:
   q_t + F(q)_x = 0
   F(q) = umax * q * (1 - q)
Here q(t,x) is the density of traffic and U(q) = umax * (1 - q) is the velocity

The local cell-face flux problem uses equation (12.4) on page 229 of [1], the
Godunov flux function.

The boundary conditions are either outflow or Dirichlet, and the initial
conditions are assigned, according to the three particular problems which are
shown in Figure 13.1 ("bulge"), Figure 13.2 ("redlight"), and Figure 13.3
("greenlight").

See cases.h and riemann.c for the full solver.

[1] R. LeVeque, "Finite Volume Methods for Hyperbolic Problems", Cambridge
    University Press, 2002.
*/

// values from:  https://depts.washington.edu/clawpack/clawpack-4.3/
static const PetscReal traffic_umax = 1.0,       // velocity = umax * (1 - q)
                       traffic_alpha = 0.7,      // used in bulge; this is a guess
                       traffic_beta = 0.01,      // used in bulge
                       traffic_qleft = 0.25,     // used in bulge, redlight
                       traffic_qright = 1.00;    // used in redlight

static const char traffic_uname[50] = "q  (traffic density)";

// this initial condition is shown in Figure 11.1 on page 205 of LeVeque
static PetscErrorCode traffic_f_bulge(PetscReal t, PetscReal x, PetscReal *q) {
    q[0] = traffic_qleft + traffic_alpha * PetscExpReal(- traffic_beta * x * x);
    return 0;
}

// this initial condition is shown in Figure 11.3 on page 207 of LeVeque
static PetscErrorCode traffic_f_redlight(PetscReal t, PetscReal x, PetscReal *q) {
    q[0] = (x < 0.0) ? traffic_qleft : traffic_qright;
    return 0;
}

// this initial condition is shown in Figure 11.3 on page 207 of LeVeque
static PetscErrorCode traffic_f_greenlight(PetscReal t, PetscReal x, PetscReal *q) {
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

static PetscErrorCode traffic_bdryflux_a_outflow(PetscReal t, PetscReal *qr, PetscReal *F) {
    const PetscReal  s = traffic_umax * (1.0 - 2.0 * qr[0]);
    if (s <= 0.0)
        F[0] = traffic_umax * qr[0] * (1.0 - qr[0]);
    else {
        SETERRQ1(PETSC_COMM_SELF,1,
                 "assumption of outflow s <= 0.0 fails at (t=%g,a)\n",t);
    }
    return 0;
}

static PetscErrorCode traffic_bdryflux_a_dirichlet(PetscReal t, PetscReal *qr, PetscReal *F) {
    F[0] = traffic_umax * traffic_qleft * (1.0 - traffic_qleft);
    return 0;
}

static PetscErrorCode traffic_bdryflux_b_outflow(PetscReal t, PetscReal *ql, PetscReal *F) {
    const PetscReal  s = traffic_umax * (1.0 - 2.0 * ql[0]);
    if (s >= 0.0)
        F[0] = traffic_umax * ql[0] * (1.0 - ql[0]);
    else {
        SETERRQ1(PETSC_COMM_SELF,1,
                 "assumption of outflow s >= 0 fails at (t=%g,b)\n",t);
    }
    return 0;
}

static PetscErrorCode traffic_bdryflux_b_dirichlet(PetscReal t, PetscReal *qr, PetscReal *F) {
    F[0] = traffic_umax * traffic_qright * (1.0 - traffic_qright);
    return 0;
}

// compute maximum speed (of a characteristic) from u; note F'(q) = umax (1 - 2 q)
static PetscErrorCode traffic_maxspeed(PetscReal t, PetscReal x, PetscReal *q,
                                       PetscReal *speed) {
    // see equation (11.16) regarding speed of characteristic
    *speed = PetscAbs(traffic_umax * (1.0 - 2.0 * q[0]));
    return 0;
}

typedef enum {TR_BULGE,TR_REDLIGHT,TR_GREENLIGHT} TRInitialType;
static const char* TRInitialTypes[] = {"bulge","redlight","greenlight",
                                       "TRInitialType", "", NULL};

PetscErrorCode  TrafficInitializer(ProblemCtx *user) {
    PetscErrorCode  ierr;
    TRInitialType   initial = TR_BULGE;

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
    user->t0_default = 0.0;
    user->periodic_bcs = PETSC_FALSE;
    user->g_source = &traffic_g;
    user->faceflux = &traffic_faceflux;
    user->maxspeed = &traffic_maxspeed;

    switch (initial) {
        case TR_BULGE:  // values for Figure 11.1 in LeVeque
            user->a_left = -30.0;
            user->b_right = 30.0;
            user->tf_default = 25.0;
            user->f_initial = &traffic_f_bulge;
            user->bdryflux_a = &traffic_bdryflux_a_dirichlet;
            user->bdryflux_b = &traffic_bdryflux_b_outflow;
            break;
        case TR_REDLIGHT:  // values for Figure 11.2 in LeVeque
            user->a_left = -40.0;
            user->b_right = 10.0;
            user->tf_default = 36.0;
            user->f_initial = &traffic_f_redlight;
            user->bdryflux_a = &traffic_bdryflux_a_dirichlet;
            user->bdryflux_b = &traffic_bdryflux_b_dirichlet;
            break;
        case TR_GREENLIGHT:  // values for Figure 11.3 in LeVeque
            user->a_left = -30.0;
            user->b_right = 20.0;
            user->tf_default = 18.0;
            user->f_initial = &traffic_f_greenlight;
            user->bdryflux_a = &traffic_bdryflux_a_outflow;
            user->bdryflux_b = &traffic_bdryflux_b_outflow;
            break;
    }

    return 0;
}

