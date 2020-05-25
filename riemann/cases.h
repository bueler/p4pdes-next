#ifndef CASES_H_
#define CASES_H_

#include "acoustic.h"

typedef struct {
    PetscInt        n_dim;
    char            **fieldnames;  // FIXME  to use with DMDASetFieldNames()
    PetscReal       a_left, b_right;
    PetscErrorCode  (*f_initial)(PetscReal t, PetscReal x,
                                 PetscReal *q);
    PetscErrorCode  (*g_source)(PetscReal t, PetscReal x, PetscReal *q,
                                PetscReal *g);
    PetscErrorCode  (*PhiL_bdryflux)(PetscReal t, PetscReal *qr,
                                     PetscReal *F);
    PetscErrorCode  (*PhiR_bdryflux)(PetscReal t, PetscReal *ql,
                                     PetscReal *F);
    PetscErrorCode  (*faceflux)(PetscReal t, PetscReal x, PetscReal *ql, PetscReal *qr,
                                PetscReal *F);
} ProblemCtx;

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
    } else {
        SETERRQ(PETSC_COMM_SELF,1,"unknown problem_number\n");
    }
    return 0;
}

PetscErrorCode  DestroyCase(ProblemCtx *user) {
    //FIXME
    return 0;
}

#endif

