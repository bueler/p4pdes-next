#ifndef SWATER_H_
#define SWATER_H_

/* for constants see setprob.data and qinit.f in download from
http://depts.washington.edu/clawpack/clawpack-4.3/book/chap13/swhump1/www/index.html
thus:
grav = 1.0
beta = 5.0
q0(x) = 1.0 + 0.4 exp(-beta x^2)
q1(x) = 0.0
*/

/*
This problem is documented in section 13.1 of LeVeque*:
    h_t + (u h)_x = 0
    (h u)_t + (h u^2 + (g/2) h^2)_x = 0

FIXME  JUST A COPY OF ACOUSTIC FROM HERE
where p(t,x) is pressure and u(t,x) is velocity.  The parameter and boundary
values come from pages 50--51 and 61.  See Figures 3.1 and 3.8.  The flux is
    F(q) = [K0 u;
            (1/rho0) p]
and so
    F'(q) = A = [0,       K0;
                (1/rho0), 0]
is constant.  The boundary conditions are a closed (reflecting end) on the
left and outflow (no reflection) on the right end.

* R. LeVeque, "Finite Volume Methods for Hyperbolic Problems", Cambridge
  University Press, 2002.
*/

const PetscInt  acoustic_n = 2;
const PetscReal acoustic_K0 = 0.25,
                acoustic_rho0 = 1.0,
                acoustic_a = -1.0,
                acoustic_b = 1.0,
                acoustic_t0 = 0.0,
                acoustic_tf = 1.0;

PetscErrorCode acoustic_f(PetscReal t, PetscReal x, PetscReal *q) {
    q[0] = 0.5 * PetscExpReal(-80.0*x*x);
    if (x > -0.3 && x < -0.1)
        q[0] += 0.5;
    q[1] = 0.0;
    return 0;
}

PetscErrorCode acoustic_g(PetscReal t, PetscReal x, PetscReal *q, PetscReal *g) {
    PetscInt l;
    for (l = 0; l < acoustic_n; l++)
        g[l] = 0.0;
    return 0;
}

// closed end boundary condition:  u(t,a) = 0
PetscErrorCode acoustic_bdryflux_a(PetscReal t, PetscReal *qr, PetscReal *F) {
    const PetscReal  pa = qr[0] - qr[1],  // p(t,a) = p(t_n,x_0) - u(t_n,x_0)
                     ua = 0.0;            // u(t,a) = 0
    F[0] = acoustic_K0 * ua;
    F[1] = (1.0/acoustic_rho0) * pa;
    return 0;
}

// reflecting boundary condition:  p(t,b) - Z0 u(t,b) = 0
PetscErrorCode acoustic_bdryflux_b(PetscReal t, PetscReal *ql, PetscReal *F) {
    const PetscReal  Z0 = PetscSqrtReal(acoustic_K0 * acoustic_rho0),
                     tmp = ql[0] + ql[1],
                     pb = Z0 * tmp / (Z0 + 1.0),
                     ub = tmp / (Z0 + 1.0);
    F[0] = acoustic_K0 * ub;
    F[1] = (1.0/acoustic_rho0) * pb;
    return 0;
}

// compute flux at internal faces from left and right values of q = [p, u]
//FIXME  here is where all the action is ... needs clear documentation
PetscErrorCode acoustic_faceflux(PetscReal t, PetscReal x,
       PetscReal *ql, PetscReal *qr, PetscReal *F) {
    const PetscReal  c0 = (ql[0] + ql[1]) / 2.0,
                     c1 = (qr[0] - qr[1]) / 2.0,
                     pface = c0 + c1,
                     uface = c0 - c1;
    F[0] = acoustic_K0 * uface;
    F[1] = (1.0/acoustic_rho0) * pface;
    return 0;
}

#endif

