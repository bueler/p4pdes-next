#ifndef CASES_H_
#define CASES_H_


typedef struct {
    PetscInt        n_dim;
    char            **fieldnames;  // FIXME  to use with DMDASetFieldNames()
    PetscReal       a_left, b_right;
    PetscErrorCode  (*f_initial)(PetscReal, PetscReal, PetscReal*);
    PetscErrorCode  (*g_source)(PetscReal, PetscReal, PetscReal*, PetscReal*);
    PetscErrorCode  (*PhiL_bdryflux)(PetscReal, PetscReal*);
    PetscErrorCode  (*PhiR_bdryflux)(PetscReal, PetscReal*);
    PetscErrorCode  (*faceflux)(PetscReal, PetscReal, PetscReal*, PetscReal*, PetscReal*);
} ProblemCtx;


// ********** acoustic **********
const PetscInt  acoustic_n = 2;
const PetscReal acoustic_K = 1.0,   //FIXME  match LeVeque problem
                acoustic_p0 = 1.0,  //FIXME
                acoustic_a = 0.0,  //FIXME
                acoustic_b = 1.0;

PetscErrorCode acoustic_f(PetscReal t, PetscReal x, PetscReal *u) {
    if (x > 0.4 && x < 0.6)   //FIXME  match LeVeque problem
        u[0] = 1.0;
    else
        u[0] = 0.0;
    u[1] = 0.0;
    return 0;
}

PetscErrorCode acoustic_g(PetscReal t, PetscReal x, PetscReal *u, PetscReal *b) {
    PetscInt l;
    for (l = 0; l < acoustic_n; l++)
        b[l] = 0.0;
    return 0;
}

PetscErrorCode acoustic_PhiL(PetscReal t, PetscReal *F) {   //FIXME  match LeVeque problem
    PetscInt l;
    for (l = 0; l < acoustic_n; l++)
        F[l] = 0.0;
    return 0;
}

PetscErrorCode acoustic_PhiR(PetscReal t, PetscReal *F) {   //FIXME  match LeVeque problem
    PetscInt l;
    for (l = 0; l < acoustic_n; l++)
        F[l] = 0.0;
    return 0;
}

// here is where all the action is ... needs clear documentation
PetscErrorCode acoustic_faceflux(PetscReal t, PetscReal x,
       PetscReal *ul, PetscReal *ur, PetscReal *F) {   //FIXME  rethink eigenvector calc
    PetscReal      c = 1.0,   //FIXME  match LeVeque problem; use K,p0
                   c0 = (ul[0] + ul[1]) / 2.0,
                   c1 = (ur[0] - ur[1]) / 2.0;
    // uface[0] = c0 + c1;
    // uface[1] = c0 - c1;
    // F = A uface
    F[0] = c * (c0 - c1);
    F[1] = c * (c0 + c1);
    return 0;
}
// **********



PetscErrorCode  CreateCase(PetscInt problem_number, ProblemCtx *user) {
    user->fieldnames = NULL;
    if (problem_number == 0) {  // FIXME problem_number clunky
        user->n_dim = acoustic_n;
        user->a_left = acoustic_a;
        user->b_right = acoustic_b;
        user->f_initial = &acoustic_f;
        user->g_source = &acoustic_g;
        user->PhiL_bdryflux = &acoustic_PhiL;
        user->PhiR_bdryflux = &acoustic_PhiR;
        user->faceflux = &acoustic_faceflux;
    }
    return 0;
}

PetscErrorCode  DestroyCase(ProblemCtx *user) {
    //FIXME
    return 0;
}

#endif

