/*
This problem is equation (2.50) in LeVeque [1]:
    p_t + K0 u_x = 0
    u_t + (1/rho0) p_x = 0
where p(t,x) is pressure and u(t,x) is velocity.  The parameter and boundary
values come from pages 50--51 and 61.  See Figures 3.1 and 3.8.  The flux is
    F(q) = [K0 u;
            (1/rho0) p]
and so
    F'(q) = A = [0,       K0;
                (1/rho0), 0]
is constant.  The boundary conditions are a closed (reflecting end) on the
left and outflow (no reflection) on the right end.

The (ordered) left eigenvector expansion of A is
    lambda0 = - c0,   w0 = [1, - Z0]
    lambda1 = + c0,   w1 = [1, + Z0]
where  c0 = sqrt(K0/rho0) and Z0 = rho0 c0, thus the Riemann invariants are
    v0(t,x) = w0' * q = p(t,x) - Z0 u(t,x)    going left
    v1(t,x) = w1' * q = p(t,x) + Z0 u(t,x)    going right

The local cell-face flux problem uses
    pface(t,xface) = (1/2) * (v0(tn,xright) + v1(tn,xleft))
    uface(t,xface) = (1/(2Z0)) * (v0(tn,xright) - v1(tn,xleft))
to give
    faceflux(ql,qr) = [K0 uface;
                       (1/rho0) pface]

The boundary conditions need specific arguments for closed end and outflow
cases.

[1] R. LeVeque, "Finite Volume Methods for Hyperbolic Problems", Cambridge
    University Press, 2002.
*/

// prefix acoustic_ is essentially a namespace

static const PetscReal acoustic_K0 = 0.25,
                       acoustic_rho0 = 1.0;
static PetscReal       acoustic_Z0;   // see AcousticInitializer()

static const char acoustic_pname[50] = "p (pressure)",
                  acoustic_uname[50] = "u (velocity)";

// this initial condition is on pages 50--51 of LeVeque
static PetscErrorCode acoustic_leveque(PetscReal t, PetscReal x, PetscReal *q) {
    q[0] = 0.5 * PetscExpReal(-80.0*x*x);
    if (x > -0.3 && x < -0.1)
        q[0] += 0.5;
    q[1] = 0.0;
    return 0;
}

static PetscErrorCode acoustic_stump(PetscReal t, PetscReal x, PetscReal *q) {
    q[0] = (x > -0.3 && x < 0.3) ? 1.0 : 0.0;
    q[1] = 0.0;
    return 0;
}

// no source term
static PetscErrorCode acoustic_g(PetscReal t, PetscReal x, PetscReal *q, PetscReal *g) {
    g[0] = 0.0;
    g[1] = 0.0;
    return 0;
}

// evaluate F(q)
static inline void acoustic_evalflux(PetscReal p, PetscReal u, PetscReal *F) {
    F[0] = acoustic_K0 * u;
    F[1] = (1.0/acoustic_rho0) * p;
}

// closed end boundary condition:  u(t,a) = 0
static PetscErrorCode acoustic_bdryflux_a(PetscReal t, PetscReal *qr, PetscReal *F) {
    const PetscReal  pa = qr[0] - acoustic_Z0 * qr[1], // p(t,a) = p(t_n,x_0) - Z0 u(t_n,x_0)
                     ua = 0.0;                         // u(t,a) = 0
    acoustic_evalflux(pa,ua,F);
    return 0;
}

// outflow boundary condition:  p(t,b) - Z0 u(t,b) = 0  (zero the incoming Riemann invariant)
static PetscErrorCode acoustic_bdryflux_b(PetscReal t, PetscReal *ql, PetscReal *F) {
    const PetscReal  tmp = ql[0] + acoustic_Z0 * ql[1],
                     pb = 0.5 * tmp,
                     ub = 0.5 * tmp / acoustic_Z0;
    acoustic_evalflux(pb,ub,F);
    return 0;
}

// compute flux at internal faces from left and right values of q = [p, u]
static PetscErrorCode acoustic_faceflux(PetscReal t, PetscReal x,
       PetscReal *ql, PetscReal *qr, PetscReal *F) {
    const PetscReal  pface = 0.5 * (qr[0] + ql[0] + acoustic_Z0 * (ql[1] - qr[1])),
                     uface = 0.5 * ((1.0/acoustic_Z0) * (ql[0] - qr[0]) + ql[1] + qr[1]);
    acoustic_evalflux(pface,uface,F);
    return 0;
}

// return c0
static PetscErrorCode acoustic_maxspeed(PetscReal t, PetscReal x, PetscReal *q,
                                        PetscReal *speed) {
    *speed = PetscSqrtReal(acoustic_K0 / acoustic_rho0);
    return 0;
}

typedef enum {AC_LEVEQUE,AC_STUMP} AcousticInitialType;
static const char* AcousticInitialTypes[] = {"leveque","stump",
                                             "AcousticInitialType", "", NULL};

PetscErrorCode  AcousticInitializer(ProblemCtx *user) {
    PetscErrorCode       ierr;
    AcousticInitialType  initial = AC_LEVEQUE;

    if (user == NULL) {
        SETERRQ(PETSC_COMM_SELF,1,"ProblemCtx *user is NULL\n");
    }

    ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"",
               "options for acoustic solver (-problem acoustic)",""); CHKERRQ(ierr);
    ierr = PetscOptionsEnum("-initial", "acoustic initial condition",
               "riemann.c",AcousticInitialTypes,(PetscEnum)(initial),(PetscEnum*)&initial,
               NULL); CHKERRQ(ierr);
    ierr = PetscOptionsEnd(); CHKERRQ(ierr);

    user->n_dim = 2;
    ierr = PetscMalloc1(2,&(user->field_names)); CHKERRQ(ierr);
    (user->field_names)[0] = (char*)acoustic_pname;
    (user->field_names)[1] = (char*)acoustic_uname;
    user->a_left = -1.0;
    user->b_right = 1.0;
    user->t0_default = 0.0;
    user->tf_default = 1.0;
    user->periodic_bcs = PETSC_FALSE;
    user->f_initial = (initial == AC_LEVEQUE) ? &acoustic_leveque : &acoustic_stump;
    user->g_source = &acoustic_g;
    user->bdryflux_a = &acoustic_bdryflux_a;
    user->bdryflux_b = &acoustic_bdryflux_b;
    user->faceflux = &acoustic_faceflux;
    user->maxspeed = &acoustic_maxspeed;
    acoustic_Z0 = PetscSqrtReal(acoustic_K0 * acoustic_rho0);
    return 0;
}

