static char help[] =
"DMPlex mesh test suggested by Matt Knepley on petsc-users 11/9/21.\n"
"Shows DMPlex is more capable at the command line than I thought.\n\n";

/*
running with the -dm_view hdf5 seems to require a library I don't have:
mpiexec -n 2 ./meshtest -dm_plex_shape sphere -dm_refine_pre 3 -dm_extrude 5 -dm_plex_transform_extrude_thickness 0.1 -dm_distribute -dm_view hdf5:mesh.h5

this runs without complaint:
mpiexec -n 2 ./meshtest -dm_plex_shape sphere -dm_refine_pre 3 -dm_extrude 5 -dm_plex_transform_extrude_thickness 0.1 -dm_distribute -dm_view
*/

#include <petsc.h>

int main(int argc, char **argv)
{
  DM             dm;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);  if (ierr) return ierr;
  ierr = DMCreate(PETSC_COMM_WORLD, &dm); CHKERRQ(ierr);
  ierr = DMSetType(dm, DMPLEX); CHKERRQ(ierr);
  ierr = DMSetFromOptions(dm); CHKERRQ(ierr);
  ierr = DMViewFromOptions(dm, NULL, "-dm_view"); CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  return PetscFinalize();
}