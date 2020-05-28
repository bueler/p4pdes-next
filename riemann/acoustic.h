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

const PetscReal acoustic_K0 = 0.25,
                acoustic_rho0 = 1.0;
PetscReal       acoustic_Z0;   // see AcousticInitializer()

const char acoustic_pname[50] = "p (pressure)",
           acoustic_uname[50] = "u (velocity)";

// this initial condition is on pages 50--51 of LeVeque
PetscErrorCode acoustic_f(PetscReal t, PetscReal x, PetscReal *q) {
    q[0] = 0.5 * PetscExpReal(-80.0*x*x);
    if (x > -0.3 && x < -0.1)
        q[0] += 0.5;
    q[1] = 0.0;
    return 0;
}

// no source term
PetscErrorCode acoustic_g(PetscReal t, PetscReal x, PetscReal *q, PetscReal *g) {
    g[0] = 0.0;
    g[1] = 0.0;
    return 0;
}

// closed end boundary condition:  u(t,a) = 0
PetscErrorCode acoustic_bdryflux_a(PetscReal t, PetscReal *qr, PetscReal *F) {
    const PetscReal  pa = qr[0] - acoustic_Z0 * qr[1], // p(t,a) = p(t_n,x_0) - Z0 u(t_n,x_0)
                     ua = 0.0;                         // u(t,a) = 0
    F[0] = acoustic_K0 * ua;
    F[1] = (1.0/acoustic_rho0) * pa;
    return 0;
}

// outflow boundary condition:  p(t,b) - Z0 u(t,b) = 0  (zero the incoming Riemann invariant)
PetscErrorCode acoustic_bdryflux_b(PetscReal t, PetscReal *ql, PetscReal *F) {
    const PetscReal  tmp = ql[0] + acoustic_Z0 * ql[1],
                     pb = 0.5 * tmp,
                     ub = 0.5 * tmp / acoustic_Z0;
    F[0] = acoustic_K0 * ub;
    F[1] = (1.0/acoustic_rho0) * pb;
    return 0;
}

// compute flux at internal faces from left and right values of q = [p, u]
PetscErrorCode acoustic_faceflux(PetscReal t, PetscReal x,
       PetscReal *ql, PetscReal *qr, PetscReal *F) {
    const PetscReal  pface = 0.5 * (qr[0] + ql[0] + acoustic_Z0 * (ql[1] - qr[1])),
                     uface = 0.5 * ((1.0/acoustic_Z0) * (ql[0] - qr[0]) + ql[1] + qr[1]);
    F[0] = acoustic_K0 * uface;
    F[1] = (1.0/acoustic_rho0) * pface;
    return 0;
}

PetscErrorCode  AcousticInitializer(ProblemCtx *user) {
    PetscErrorCode  ierr;

    if (user == NULL) {
        SETERRQ(PETSC_COMM_SELF,1,"ProblemCtx *user is NULL\n");
    }
    user->n_dim = 2;
    ierr = PetscMalloc1(2,&(user->field_names)); CHKERRQ(ierr);
    (user->field_names)[0] = (char*)acoustic_pname;
    (user->field_names)[1] = (char*)acoustic_uname;
    user->a_left = -1.0;
    user->b_right = 1.0;
    user->t0_default = 0.0;
    user->tf_default = 1.0;
    user->f_initial = &acoustic_f;
    user->g_source = &acoustic_g;
    user->bdryflux_a = &acoustic_bdryflux_a;
    user->bdryflux_b = &acoustic_bdryflux_b;
    user->faceflux = &acoustic_faceflux;
    acoustic_Z0 = PetscSqrtReal(acoustic_K0 * acoustic_rho0);
    return 0;
}

