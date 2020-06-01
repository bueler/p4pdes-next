/*
FIXME document simple advection equation  u_t + a_0 u_x = 0
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

// compute flux at internal faces from left and right values of q = [u]
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
    PetscErrorCode  ierr;

    if (user == NULL) {
        SETERRQ(PETSC_COMM_SELF,1,"ProblemCtx *user is NULL\n");
    }

    ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"",
               "options for advection solver (-problem advection)",""); CHKERRQ(ierr);
    ierr = PetscOptionsReal("-a", "constant transport velocity",
               "riemann.c",advection_a,&advection_a,NULL); CHKERRQ(ierr);
    ierr = PetscOptionsEnd(); CHKERRQ(ierr);

    user->n_dim = 1;
    ierr = PetscMalloc1(1,&(user->field_names)); CHKERRQ(ierr);
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

