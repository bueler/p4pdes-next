/*
This is the simple scalar, linear advection equation
   u_t + a u_x = 0
where a is constant (and of either sign).  The solution u(t,x), which may be
interpreted as a passive tracer, is known to be
   u(t,x) = f(x - a t)
if u(0,x) = f(x) is the initial condition.  The boundary conditions are
periodic.

The local cell-face flux problem uses the Godunov method, which is to say
simple first-order upwinding.  However, this is modified by slope-limiting
in the riemann.c solver code.

The initial condition is seen in Figures 6.1 and 6.2 of LeVeque [1].

See cases.h and riemann.c for the full solver.

[1] R. LeVeque, "Finite Volume Methods for Hyperbolic Problems", Cambridge
    University Press, 2002.
*/

// values from:
// https://depts.washington.edu/clawpack/users/claw/book/chap6/compareadv/setprob.data
static PetscReal advection_a = 1.0,
                 advection_beta = 200;

static const char advection_uname[50] = "u";

// this initial condition is shown in Figure 6.1 on page 101 of LeVeque
static PetscErrorCode advection_f(PetscReal t, PetscReal x, PetscReal *q) {
    q[0] = PetscExpReal(-advection_beta * (x - 0.3)*(x - 0.3));
    q[0] += (x > 0.6 && x < 0.8) ? 1.0 : 0.0;
    return 0;
}

// no source term
static PetscErrorCode advection_g(PetscReal t, PetscReal x, PetscReal *q, PetscReal *g) {
    g[0] = 0.0;
    return 0;
}

// compute flux at internal faces from left and right values of q = [u];
// straightforward first-order upwinding
static PetscErrorCode advection_faceflux(PetscReal t, PetscReal x,
       PetscReal *ql, PetscReal *qr, PetscReal *F) {
    F[0] = advection_a * ((advection_a >= 0.0) ? ql[0] : qr[0]);
    return 0;
}

// return c0
static PetscErrorCode advection_maxspeed(PetscReal t, PetscReal x, PetscReal *q,
                                        PetscReal *speed) {
    *speed = advection_a;
    return 0;
}

PetscErrorCode  AdvectionInitializer(ProblemCtx *user) {
    if (user == NULL) {
        SETERRQ(PETSC_COMM_SELF,1,"ProblemCtx *user is NULL\n");
    }

    PetscOptionsBegin(PETSC_COMM_WORLD,"",
               "options for advection solver (riemann -problem advection)","");
    PetscCall(PetscOptionsReal("-a", "constant transport velocity",
               "advection.h",advection_a,&advection_a,NULL));
    PetscOptionsEnd();

    user->n_dim = 1;
    PetscCall(PetscMalloc1(1,&(user->field_names)));
    (user->field_names)[0] = (char*)advection_uname;
    user->a_left = 0.0;
    user->b_right = 1.0;
    user->t0_default = 0.0;
    user->tf_default = 1.0;
    user->periodic_bcs = PETSC_TRUE;
    user->f_initial = &advection_f;
    user->g_source = &advection_g;
    user->bdryflux_a = NULL;  // periodic b.c.s
    user->bdryflux_b = NULL;
    user->faceflux = &advection_faceflux;
    user->maxspeed = &advection_maxspeed;
    return 0;
}

