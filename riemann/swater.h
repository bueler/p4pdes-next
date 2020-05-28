/* FIXME  document eigenvectors, Riemann invariants, local face flux computation from Riemann solver

from LeVeque [1] pages 254--257; specific problem is for Figure 13.1

use Roe solver designed for shallow water, section 15.3.3

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

PetscErrorCode swater_f(PetscReal t, PetscReal x, PetscReal *q) {
    q[0] = 1.0 + 0.4 * PetscExpReal(- 5.0 * x*x);
    q[1] = 0.0;
    return 0;
}

PetscErrorCode swater_g(PetscReal t, PetscReal x, PetscReal *q, PetscReal *g) {
    g[0] = 0.0;
    g[1] = 0.0;
    return 0;
}

// FIXME want outflow boundary condition at x=a
PetscErrorCode swater_bdryflux_a(PetscReal t, PetscReal *qr, PetscReal *F) {
    F[0] = 0.0;
    F[1] = 0.0;
    return 0;
}

// FIXME want outflow boundary condition at x=b
PetscErrorCode swater_bdryflux_b(PetscReal t, PetscReal *ql, PetscReal *F) {
    F[0] = 0.0;
    F[1] = 0.0;
    return 0;
}


// compute flux at internal faces from left and right values of q = [h, h u]
//FIXME
PetscErrorCode swater_faceflux(PetscReal t, PetscReal x,
       PetscReal *ql, PetscReal *qr, PetscReal *F) {
    F[0] = 0.0;
    F[1] = 0.0;
    return 0;
}


PetscErrorCode  SWaterInitializer(ProblemCtx *user) {
    PetscErrorCode  ierr;

    if (user == NULL) {
        SETERRQ(PETSC_COMM_SELF,1,"ProblemCtx *user is NULL\n");
    }
    user->n_dim = 2;
    ierr = PetscMalloc1(2,&(user->field_names)); CHKERRQ(ierr);
    (user->field_names)[0] = (char*)swater_hname;
    (user->field_names)[1] = (char*)swater_huname;
    user->a_left = -5.0;
    user->b_right = 5.0;
    user->t0_default = 0.0;
    user->tf_default = 3.0;
    user->f_initial = &swater_f;
    user->g_source = &swater_g;
    user->bdryflux_a = &swater_bdryflux_a;
    user->bdryflux_b = &swater_bdryflux_b;
    user->faceflux = &swater_faceflux;
    return 0;
}
