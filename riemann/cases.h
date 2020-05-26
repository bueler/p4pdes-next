#ifndef CASES_H_
#define CASES_H_

/* To add a new problem create  NEWPROB.h  by copying acoustic.h or etc.
and editing.  Include it in the #include list below.  Then edit CreateCase()
for the new problem_number.  Then adjust options in riemann.c as needed. */

#include "acoustic.h"
//#include "swater.h"  FIXME

typedef struct {
    PetscInt        n_dim;
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


typedef enum {ACOUSTIC, SWATER} ProblemType;
static const char* ProblemTypes[] = {"acoustic","swater",
                                     "ProblemType", "", NULL};


PetscErrorCode  CreateCase(PetscInt problem_number, ProblemType *problem, ProblemCtx *user) {
    PetscErrorCode  ierr;
    PetscInt        k;

    *problem = ACOUSTIC;
    ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"",
               "riemann (hyperbolic system solver) options",""); CHKERRQ(ierr);
    ierr = PetscOptionsEnum("-problem", "problem type",
               "riemann.c",ProblemTypes,(PetscEnum)(*problem),(PetscEnum*)problem,
               NULL); CHKERRQ(ierr);
    ierr = PetscOptionsEnd(); CHKERRQ(ierr);

    switch (*problem) {
        case ACOUSTIC:
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
            break;
        case SWATER:
            SETERRQ(PETSC_COMM_SELF,1,"problem swater not implemented\n");
            break;
        default:
            SETERRQ(PETSC_COMM_SELF,2,"unknown problem_number\n");
    }

    ierr = PetscMalloc1(user->n_dim,&(user->field_names)); CHKERRQ(ierr);
    for (k = 0; k < user->n_dim; k++) {
       ierr = PetscMalloc1(50,&((user->field_names)[k])); CHKERRQ(ierr);
    }

    switch (*problem) {
        case ACOUSTIC:
            strcpy((user->field_names)[0],"p (pressure)");
            strcpy((user->field_names)[1],"u (velocity)");
            break;
        case SWATER:
            strcpy((user->field_names)[0],"h (surface height)");
            strcpy((user->field_names)[1],"h u (height times velocity)");
            break;
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

