// the following precomplier directives are needed to avoid reading this
// header file multiple times
#ifndef CASES_H_
#define CASES_H_

/* To add a new problem create  NEWPROBLEM.h  by copying acoustic.h or etc.
and editing.  Include it in the #include list at the bottom.  Then edit
ProblemType, ProblemTypes, InitializerPtrs in the BLOCK below to include it. */

typedef struct {
    PetscInt        n_dim;
    char            **field_names;
    PetscReal       a_left, b_right,
                    t0_default, tf_default;
    PetscBool       periodic_bcs;
    PetscErrorCode  (*f_initial)(PetscReal t, PetscReal x,
                                 PetscReal *q);
    PetscErrorCode  (*g_source)(PetscReal t, PetscReal x, PetscReal *q,
                                PetscReal *g);
    PetscErrorCode  (*bdryflux_a)(PetscReal t, PetscReal hx, PetscReal *qr,
                                  PetscReal *F);
    PetscErrorCode  (*bdryflux_b)(PetscReal t, PetscReal hx, PetscReal *ql,
                                  PetscReal *F);
    PetscErrorCode  (*faceflux)(PetscReal t, PetscReal x, PetscReal *ql, PetscReal *qr,
                                PetscReal *F);
    PetscErrorCode  (*maxspeed)(PetscReal t, PetscReal x, PetscReal *q,
                                PetscReal *speed);
} ProblemCtx;

typedef PetscErrorCode ProblemInitializer(ProblemCtx*);


/* To add a new problem make edits in the following BLOCK. */
#include "acoustic.h"
#include "advection.h"
#include "swater.h"
#include "traffic.h"
typedef enum {ACOUSTIC,
              ADVECTION,
              SWATER,
              TRAFFIC} ProblemType;
static const char* ProblemTypes[] = {"acoustic",
                                     "advection",
                                     "swater",
                                     "traffic",
                                     "ProblemType", "", NULL};
static ProblemInitializer* InitializerPtrs[] = {&AcousticInitializer,
                                                &AdvectionInitializer,
                                                &SWaterInitializer,
                                                &TrafficInitializer};
/* end BLOCK */

#endif
