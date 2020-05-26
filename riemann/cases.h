#ifndef CASES_H_
#define CASES_H_

/* To add a new problem create  NEWPROB.h  by copying acoustic.h (or other)
and editing.  Include it in the #include list below.  Then edit CreateCase()
for the new problem_number.  Then adjust options in riemann.c as needed. */

#include "acoustic.h"

typedef struct {
    PetscInt        n_dim;
    char            problem_name[50];
    char            **field_names;
    PetscReal       a_left, b_right,
                    t0_default, tf_default;
    PetscErrorCode  (*f_initial)(PetscReal t, PetscReal x,
                                 PetscReal *q);
    PetscErrorCode  (*g_source)(PetscReal t, PetscReal x, PetscReal *q,
                                PetscReal *g);
    PetscErrorCode  (*bdryflux_a)(PetscReal t, PetscReal *qr,
                                  PetscReal *F);
    PetscErrorCode  (*bdryflux_b)(PetscReal t, PetscReal *ql,
                                  PetscReal *F);
    PetscErrorCode  (*faceflux)(PetscReal t, PetscReal x, PetscReal *ql, PetscReal *qr,
                                PetscReal *F);
} ProblemCtx;

PetscErrorCode  CreateCase(PetscInt problem_number, ProblemCtx *user) {
    PetscErrorCode  ierr;
    PetscInt        k;
    if (problem_number == 0) {  // FIXME problem_number clunky
        strcpy(user->problem_name,"acoustic");
        user->n_dim = acoustic_n;
        user->a_left = acoustic_a;
        user->b_right = acoustic_b;
        user->t0_default = acoustic_t0;
        user->tf_default = acoustic_tf;
        user->f_initial = &acoustic_f;
        user->g_source = &acoustic_g;
        user->bdryflux_a = &acoustic_bdryflux_a;
        user->bdryflux_b = &acoustic_bdryflux_b;
        user->faceflux = &acoustic_faceflux;
    } else {
        SETERRQ(PETSC_COMM_SELF,1,"unknown problem_number\n");
    }
    ierr = PetscMalloc1(user->n_dim,&(user->field_names)); CHKERRQ(ierr);
    for (k = 0; k < user->n_dim; k++) {
       ierr = PetscMalloc1(50,&((user->field_names)[k])); CHKERRQ(ierr);
    }
    if (problem_number == 0) {
        strcpy((user->field_names)[0],"p (pressure)");
        strcpy((user->field_names)[1],"u (velocity)");
    } else {
        SETERRQ(PETSC_COMM_SELF,2,"unknown problem_number\n");
    }
    return 0;
}

PetscErrorCode  DestroyCase(ProblemCtx *user) {
    PetscErrorCode  ierr;
    PetscInt        k;
    for (k = 0; k < user->n_dim; k++) {
       ierr = PetscFree((user->field_names)[k]); CHKERRQ(ierr);
    }
    ierr = PetscFree(user->field_names); CHKERRQ(ierr);
    return 0;
}

#endif

