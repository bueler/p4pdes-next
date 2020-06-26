/*
The one-spatial-dimension shallow water model is equation (3.1) of LeVeque,
George, Berger [2]:
    h_t + (hu)_x = 0
    (hu)_t + (hu^2 + (g/2) h^2)_x = -g h B_x
Here h(t,x) is water depth (thickness), u(t,x) is the vertically-averaged
horizontal water velocity, and B(x) is the elevation of the bathymetry.  Thus
    eta(t,x) = B(x) + h(t,x)
is the elevation of the water surface.

The constant bathymetry case (B_x = 0) is equation (13.5) in LeVeque [1]; see
pages 254--257 of [1] for a derivation of this system.  Note that the *value*
of B(x) is not used after initialization, only its derivative B_x(x).

Equivalently, if q1 = h and q2 = hu then the system is
    q1_t + q2_x = 0
    q2_t + (q2^2/q1 + (g/2) q1^2)_x = -g q1 B_x
If B_x = 0 then the system is a conservation law
    q_t + F(q)_x = 0
with flux function
    F(q) = [h;                 =  [q2;
            hu^2 + (g/2) h^2]      q2^2/q1 + (g/2) q1^2]

Thus the characteristic speeds come from the matrix
    F'(q) = [0,          1;    =  [0,                 1;
             -u^2 + g h, 2 u]      -(q2/q1)^2 + g q1, 2 (q2/q1)]

The domain is  -5 < x < 5.  All problems have nonreflecting boundary conditions
on each end.

The bathymetry is always a straight line:
    B(x) = b0 + bx x
(The default for the slope is bx = 0.)

The initial states (-initial X) and set the initial depth  h(0,x):
  X = flat:  h is set by the requirement eta(0,x) = 0
  X = hump:  problem of Figure 13.1 of [1]
  X = dam:   problem in Figure 13.4 of [1]
In each case the initial velocity is zero:  u(0,x) = 0.

The constants used here are from setprob.data and qinit.f in the download from
http://depts.washington.edu/clawpack/clawpack-4.3/book/chap13/swhump1/www/index.html
Specifically, g = 1.0 and, for the "hump" initial condition, beta = 5.0
and q0(x) = 1.0 + 0.4 exp(-beta x^2).

The Riemann solver is *not* exact.  That is, unlike in the other models solved
by riemann.c (see acoustic.h, advection.h, and traffic.h), the flux on a face
is not computed from the solution (on the face) of the problem with two levels
Q_L, Q_R in the neighboring cells.  Instead the method is the "linearized
Riemann solver" described in section 15.3.1 of LeVeque [1].  That is, we first
evaluate the 2x2 matrix
    A = F'(qhat)
where qhat is an "average" of the values Q_L, Q_R.  Then we compute the
solution of the Riemann problem for the linear problem (see acoustic.h) using
this matrix A.

But qhat is not a simple average.  Instead it is the Roe average (section 15.3.3
of LeVeque [1]).  See the source code of swater_faceflux() below.

See cases.h and riemann.c for the full solver.

This case is documented by the slides in the fvolume/ directory at
    https://github.com/bueler/slide-teach

[1] R. LeVeque, "Finite Volume Methods for Hyperbolic Problems", Cambridge
    University Press, 2002.

[2] R. LeVeque, D. George, M. Berger, "Tsunami modelling with adaptively-
    refined finite volume methods", Acta Numerica 211-289, 2011
    https://doi.org/10.1017/S0962492911000043
    http://faculty.washington.edu/rjl/pubs/AN2011/LeVequeGeorgeBerger-an11.pdf
*/

static PetscReal swater_grav = 1.0,
                 swater_b0 = -1.0,
                 swater_bx = 0.0;

static const char swater_hname[50] = "h (water depth)",
                  swater_huname[50] = "h u (depth * velocity)";

static PetscReal swater_B(PetscReal x) {
    return swater_b0 + swater_bx * x;
}

// the surface elevation is
//   eta(t,x) = B(x) + h(t,x)
// so initially flat (eta(0,x) = 0) means  h(0,x) = - B(x)
static PetscErrorCode swater_flat(PetscReal t, PetscReal x, PetscReal *q) {
    q[0] = - swater_B(x);
    q[1] = 0.0;
    return 0;
}

// see LeVeque Figure 13.1
static PetscErrorCode swater_hump(PetscReal t, PetscReal x, PetscReal *q) {
    q[0] = - swater_B(x) + 0.4 * PetscExpReal(- 5.0 * x*x);
    q[1] = 0.0;
    return 0;
}

// see LeVeque Figure 13.4
static PetscErrorCode swater_dam(PetscReal t, PetscReal x, PetscReal *q) {
    q[0] = - swater_B(x) + ((x < 0) ? 2.0 : 0.0);
    q[1] = 0.0;
    return 0;
}

static PetscErrorCode swater_g(PetscReal t, PetscReal x, PetscReal *q, PetscReal *g) {
    g[0] = 0.0;
    g[1] = - swater_grav * q[0] * swater_bx;
    return 0;
}

// evaluate F(q); for flux formula see page 255 of LeVeque
static inline void swater_evalflux(PetscReal h, PetscReal hu, PetscReal *F) {
    F[0] = hu;
    F[1] = (hu*hu / h) + (swater_grav / 2.0) * h*h;
}

// compute flux at internal faces from left and right values of q = [h, h u]
static PetscErrorCode swater_faceflux(PetscReal t, PetscReal x,
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
    // compute hbar and uhat from Roe averages, formulas (15.32) and (15.35)
    // in LeVeque
    const PetscReal hbar = 0.5 * (ql[0] + qr[0]),
                    rhl  = PetscSqrtReal(ql[0]),
                    rhr  = PetscSqrtReal(qr[0]),
                    uhat = (rhl*ql[1]/ql[0] + rhr*qr[1]/qr[0]) / (rhl + rhr),
                    delta = PetscSqrtReal(swater_grav * hbar),
                    lam0 = uhat - delta,   // note lam0 < lam1 always
                    lam1 = uhat + delta;
    // Riemann solver using lam0, lam1 speeds
    PetscReal hface, huface;
    if (lam1 <= 0.0) {          // both speeds nonpositive
        hface = qr[0];
        huface = qr[1];
    } else if (lam0 >= 0.0) {   // both speeds nonnegative
        hface = ql[0];
        huface = ql[1];
    } else {                    // lam0 < 0 < lam1
        // solve the Riemann problem with one invariant (v0) travelling left
        // and one (v1) traveling right
        const PetscReal v0r   = -lam1 * qr[0] + qr[1],
                        v1l   = -lam0 * ql[0] + ql[1];
        hface = (v1l - v0r) / (lam1 - lam0);
        huface = (lam1 * v1l - lam0 * v0r) / (lam1 - lam0);
    }
    swater_evalflux(hface,huface,F);
    return 0;
}

// Nonreflecting boundary condition at x=a: compute the boundary flux from
// zeroth-order extrapolation of Q.  This is discussed in section 7.3.1 of
// LeVeque [1] in the case of linear systems.  In particular:
//     Note that by setting [Q_{-1}=Q_0] we insure that the Riemann problem
//     at the interface [x_{-1/2}] consists of no waves, ... So in particular
//     there are no waves generated at the boundary regardless of what is
//     happening in the interior, as desired for nonreflecting boundary
//     conditions.
// With bathymetry we modify the ghost [Q_{-1}] value by the bathymetry.
static PetscErrorCode swater_bdryflux_a(PetscReal t, PetscReal hx, PetscReal *qr, PetscReal *F) {
    PetscErrorCode ierr;
    PetscReal      QL[2];
    if (qr[0] <= 0.0) {
        SETERRQ2(PETSC_COMM_SELF,1,
                 "h = qr[0] = %g is nonpositive at (t=%g,a)\n",
                 qr[0],t);
    }
    QL[0] = qr[0] + swater_bx * hx;
    if (QL[0] <= 0.0) {
        SETERRQ2(PETSC_COMM_SELF,2,
                 "ghost value h = q[-1] = %g is nonpositive at (t=%g,a)\n",
                 QL[0],t);
    }
    QL[1] = qr[1];
    ierr = swater_faceflux(t,-5.0,QL,qr,F); CHKERRQ(ierr);
    return 0;
}

// Nonreflecting boundary condition at x=b.  See comment for previous.
static PetscErrorCode swater_bdryflux_b(PetscReal t, PetscReal hx, PetscReal *ql, PetscReal *F) {
    PetscErrorCode ierr;
    PetscReal      QR[2];
    if (ql[0] <= 0.0) {
        SETERRQ2(PETSC_COMM_SELF,1,
                 "h = qr[0] = %g is nonpositive at (t=%g,b)\n",
                 ql[0],t);
    }
    QR[0] = ql[0] - swater_bx * hx;
    if (QR[0] <= 0.0) {
        SETERRQ2(PETSC_COMM_SELF,2,
                 "ghost value h = q[mx] = %g is nonpositive at (t=%g,b)\n",
                 QR[0],t);
    }
    QR[1] = ql[1];
    ierr = swater_faceflux(t,5.0,ql,QR,F); CHKERRQ(ierr);
    return 0;
}

// compute maximum speed from q using  lam_0 = u - sqrt(g h), lam_1 = u + sqrt(g h)
static PetscErrorCode swater_maxspeed(PetscReal t, PetscReal x, PetscReal *q,
                                      PetscReal *speed) {
    PetscReal delta;
    if (q[0] <= 0.0) {
        SETERRQ3(PETSC_COMM_SELF,1,
                 "h = q[0] = %g is nonpositive at (t=%g,x=%g)\n",
                 q[0],t,x);
    }
    delta = PetscSqrtReal(swater_grav * q[0]);
    *speed = PetscMax(PetscAbs(q[1] - delta),PetscAbs(q[1] + delta));
    return 0;
}


typedef enum {SW_FLAT,SW_HUMP,SW_DAM} SWInitialType;
static const char* SWInitialTypes[] = {"flat","hump","dam",
                                       "SWInitialType", "", NULL};

PetscErrorCode  SWaterInitializer(ProblemCtx *user) {
    PetscErrorCode  ierr;
    SWInitialType   initial = SW_HUMP;

    if (user == NULL) {
        SETERRQ(PETSC_COMM_SELF,1,"ProblemCtx *user is NULL\n");
    }

    ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"",
               "options for shallow water solver (riemann -problem swater)",""); CHKERRQ(ierr);
    ierr = PetscOptionsEnum("-initial", "shallow water initial condition",
               "swater.h",SWInitialTypes,(PetscEnum)(initial),(PetscEnum*)&initial,
               NULL); CHKERRQ(ierr);
    ierr = PetscOptionsReal("-b0", "bathymetry elevation at x=0",
               "swater.h",swater_b0,&swater_b0,NULL); CHKERRQ(ierr);
    ierr = PetscOptionsReal("-bx", "bathymetry slope",
               "swater.h",swater_bx,&swater_bx,NULL); CHKERRQ(ierr);
    ierr = PetscOptionsEnd(); CHKERRQ(ierr);

    user->n_dim = 2;
    ierr = PetscMalloc1(2,&(user->field_names)); CHKERRQ(ierr);
    (user->field_names)[0] = (char*)swater_hname;
    (user->field_names)[1] = (char*)swater_huname;
    user->a_left = -5.0;
    user->b_right = 5.0;
    user->t0_default = 0.0;
    user->tf_default = 3.0;
    user->periodic_bcs = PETSC_FALSE;
    switch (initial) {
        case SW_FLAT:
            user->f_initial = &swater_flat;
            break;
        case SW_HUMP:
            user->f_initial = &swater_hump;
            break;
        case SW_DAM:
            user->f_initial = &swater_dam;
            break;
    }
    user->g_source = &swater_g;
    user->bdryflux_a = &swater_bdryflux_a;
    user->bdryflux_b = &swater_bdryflux_b;
    user->faceflux = &swater_faceflux;
    user->maxspeed = &swater_maxspeed;
    return 0;
}

