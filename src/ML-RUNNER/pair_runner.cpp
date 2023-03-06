/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing authors: Albert Bartok (Cambridge University)
                         Aidan Thompson (Sandia, athomps@sandia.gov)
------------------------------------------------------------------------- */

#include "pair_runner.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "potential_file_reader.h"
#include "update.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

PairRUNNER::PairRUNNER(LAMMPS *lmp) : Pair(lmp)
{
  single_enable = 0;
  restartinfo = 0;
  one_coeff = 1;
  no_virial_fdotr_compute = 1;
  manybody_flag = 1;
  centroidstressflag = CENTROID_NOTAVAIL;
  unit_convert_flag = utils::NOCONVERT;
  map = nullptr;
  runner_potential = nullptr;
  runner_file = nullptr;
  runner_string = nullptr;
}

PairRUNNER::~PairRUNNER()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    delete[] map;
  }
  delete[] runner_potential;
  delete[] runner_file;
  delete[] runner_string;
}

void PairRUNNER::compute(int eflag, int vflag)
{
  int inum, jnum, sum_num_neigh, ii, jj, i, irunner;
  int *ilist;
  int *jlist;
  int *numneigh, **firstneigh;
  int *runner_num_neigh, *runner_neigh, *atomic_numbers;

  int nlocal = atom->nlocal;
  int nghost = atom->nghost;
  int ntotal = nlocal + nghost;
  int *type = atom->type;
  tagint *tag = atom->tag;

  double **x = atom->x;
  double **f = atom->f;

  double *runner_local_e, *runner_force, *runner_local_stress, *runner_stress, runner_energy, *lattice;

  ev_init(eflag, vflag);

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  sum_num_neigh = 0;
  runner_num_neigh = new int[inum];

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    runner_num_neigh[ii] = numneigh[i];
    sum_num_neigh += numneigh[i];
  }

  runner_neigh = new int[sum_num_neigh];
  irunner = 0;

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      runner_neigh[irunner] = (jlist[jj] & NEIGHMASK) + 1;
      irunner++;
    }
  }

  atomic_numbers = new int[ntotal];
  for (ii = 0; ii < ntotal; ii++) atomic_numbers[ii] = map[type[ii]];

  runner_local_e = new double[ntotal];
  runner_force = new double[ntotal * 3];
  runner_local_stress = new double[ntotal * 9];
  runner_stress = new double[9];

  lattice = new double[9];
  lattice[0] = domain->xprd;
  lattice[1] = 0.0;
  lattice[2] = 0.0;
  lattice[3] = domain->xy;
  lattice[4] = domain->yprd;
  lattice[5] = 0.0;
  lattice[6] = domain->xz;
  lattice[7] = domain->yz;
  lattice[8] = domain->zprd;

#if defined(LAMMPS_BIGBIG)
  int *tmptag = new int[ntotal];
  int tmplarge = 0, toolarge = 0;
  for (ii = 0; ii < ntotal; ++ii) {
    tmptag[ii] = tag[ii];
    if (tag[ii] > MAXSMALLINT) tmplarge = 1;
  }
  MPI_Allreduce(&tmplarge, &toolarge, 1, MPI_INT, MPI_MAX, world);
  if (toolarge > 0) error->all(FLERR, "Pair style runner does not support 64-bit atom IDs");

  runner_lammps_wrapper(&nlocal, &nghost, atomic_numbers, tmptag, &inum, &sum_num_neigh, ilist,
                        runner_num_neigh, runner_neigh, lattice, runner_potential, &n_runner_potential,
                        &x[0][0], &runner_energy, runner_local_e, runner_stress, runner_local_stress,
                        runner_force);

  delete[] tmptag;
#else
  runner_lammps_wrapper(&nlocal, &nghost, atomic_numbers, tag, &inum, &sum_num_neigh, ilist,
                        runner_num_neigh, runner_neigh, lattice, runner_potential, &n_runner_potential,
                        &x[0][0], &runner_energy, runner_local_e, runner_stress, runner_local_stress,
                        runner_force);
#endif

  irunner = 0;
  for (ii = 0; ii < ntotal; ii++) {
    for (jj = 0; jj < 3; jj++) {
      f[ii][jj] += runner_force[irunner];
      irunner++;
    }
  }

  if (eflag_global) { eng_vdwl = runner_energy; }

  if (eflag_atom) {
    for (ii = 0; ii < ntotal; ii++) { eatom[ii] = runner_local_e[ii]; }
  }

  if (vflag_global) {
    virial[0] = runner_stress[0];
    virial[1] = runner_stress[4];
    virial[2] = runner_stress[8];
    virial[3] = (runner_stress[3] + runner_stress[1]) * 0.5;
    virial[4] = (runner_stress[2] + runner_stress[6]) * 0.5;
    virial[5] = (runner_stress[5] + runner_stress[7]) * 0.5;
  }

  if (vflag_atom) {
    int iatom = 0;
    for (ii = 0; ii < ntotal; ii++) {
      vatom[ii][0] += runner_local_stress[iatom + 0];
      vatom[ii][1] += runner_local_stress[iatom + 4];
      vatom[ii][2] += runner_local_stress[iatom + 8];
      vatom[ii][3] += (runner_local_stress[iatom + 3] + runner_local_stress[iatom + 1]) * 0.5;
      vatom[ii][4] += (runner_local_stress[iatom + 2] + runner_local_stress[iatom + 6]) * 0.5;
      vatom[ii][5] += (runner_local_stress[iatom + 5] + runner_local_stress[iatom + 7]) * 0.5;
      iatom += 9;
    }
  }

  delete[] atomic_numbers;
  delete[] runner_num_neigh;
  delete[] runner_neigh;
  delete[] runner_local_e;
  delete[] runner_force;
  delete[] runner_stress;
  delete[] runner_local_stress;
  delete[] lattice;
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairRUNNER::settings(int narg, char ** /* arg */)
{
  if (narg != 0) error->all(FLERR, "Illegal pair_style command");

  // check if linked to the correct RUNNER library API version
  // as of 2017-07-19 this is API_VERSION 1
  if (runner_lammps_api_version() != 1)
    error->all(FLERR,
               "RUNNER LAMMPS wrapper API version is not compatible "
               "with this version of LAMMPS");

  // RUNNER potentials are parameterized in metal units // TODO Probably wrong

  if (strcmp("metal", update->unit_style) != 0)
    error->all(FLERR, "RUNNER potentials require 'metal' units");
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */
void PairRUNNER::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  setflag = memory->create(setflag, n + 1, n + 1, "pair:setflag");
  cutsq = memory->create(cutsq, n + 1, n + 1, "pair:cutsq");
  map = new int[n + 1];
}

void PairRUNNER::coeff(int narg, char **arg)
{
  if (!allocated) allocate();

  int n = atom->ntypes;
  if (narg != (4 + n))
    error->all(FLERR, "Number of arguments {} is not correct, it should be {}", narg, 4 + n);

  if (comm->me == 0) {
    PotentialFileReader reader(lmp, arg[2], "RUNNER", unit_convert_flag);
    auto comment = reader.next_string();
  }

  // use expanded file name, including LAMMPS_POTENTIALS search path
  runner_file = utils::strdup(utils::get_potential_file_path(arg[2]));
  runner_string = utils::strdup(arg[3]);
  n_runner_file = strlen(runner_file);
  n_runner_string = strlen(runner_string);

  for (int i = 4; i < narg; i++) {
    if (strcmp(arg[i], "NULL") == 0)
      map[i - 3] = -1;
    else
      map[i - 3] = utils::inumeric(FLERR, arg[i], false, lmp);
  }

  // clear setflag since coeff() called once with I,J = * *

  n = atom->ntypes;
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++) setflag[i][j] = 0;

  // set setflag i,j for type pairs where both are mapped to elements

  int count = 0;
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      if (map[i] >= 0 && map[j] >= 0) {
        setflag[i][j] = 1;
        count++;
      }

  if (count == 0) error->all(FLERR, "Incorrect args for pair coefficients");

  // Initialise potential
  // First call initializes potential via the fortran code in memory,
  // and returns the necessary size of runner_potential. This behavior
  // is invoked by setting n_potential_runner to 0.
  n_runner_potential = 0;
  runner_potential = new int[0];
  //runner_lammps_potential_initialise(runner_potential, &n_runner_potential, &cutoff, runner_file,
  //                                   &n_runner_file, runner_string, &n_runner_string);
  delete[] runner_potential;

  // Allocate runner_potential integer array. This initialise call will transfer
  // the location of the previously initialised potential to the runner_potential
  // variable, and we will use it as a handle when calling the actual calculation
  // routine. We return the cutoff as well.
  runner_potential = new int[n_runner_potential];
  //runner_lammps_potential_initialise(runner_potential, &n_runner_potential, &cutoff, runner_file,
  //                                   &n_runner_file, runner_string, &n_runner_string);
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairRUNNER::init_style()
{
  // Require newton pair on

  if (force->newton_pair != 1) error->all(FLERR, "Pair style runner requires newton pair on");

  // request full neighbor list
  neighbor->add_request(this, NeighConst::REQ_FULL);
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairRUNNER::init_one(int /*i*/, int /*j*/)
{
  return cutoff;
}
