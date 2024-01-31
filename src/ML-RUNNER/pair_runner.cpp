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
   Contributing authors: Gunnar Schmitz, Knut Nikolas Lausch

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
#include <iostream>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   constructor
------------------------------------------------------------------------- */
PairRUNNER::PairRUNNER(LAMMPS *lmp) : Pair(lmp)
{
  single_enable = 0; // HDNNP is not pairwise additive, due to three body terms
  restartinfo = 0; // do not write binary restart files
  one_coeff = 1; // parameters are read from input.nn, therefore pair_coeff only has single command
  manybody_flag = 1; // Many-body potential flag
  no_virial_fdotr_compute = 1; // ???
  centroidstressflag = CENTROID_NOTAVAIL; // ???
  unit_convert_flag = utils::NOCONVERT; // ???
  map = nullptr;

  // additional per-atom arrays for communication
  nmax = 0;
  atCharge = nullptr;
  hirshVolume = nullptr;
  elecNegativity = nullptr;
  comm_forward = 1; // forward communication (1 double per atom)
  comm_reverse = 1; // reverse communication (1 double per atom)
  commstyle = 0;
}

/* ----------------------------------------------------------------------
   destructor
------------------------------------------------------------------------- */
PairRUNNER::~PairRUNNER()
{
  // Deallocate member variables
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(atCharge);
    memory->destroy(hirshVolume);
    memory->destroy(elecNegativity);
    delete[] map;
  }
}

/* ----------------------------------------------------------------------
   main working routine
------------------------------------------------------------------------- */
void PairRUNNER::compute(int eflag, int vflag)
{
  const bool debug = false;
  bool lperiodic = true; // How to get this information in LAMMPS?

  int inum, jnum, ii, jj, i, j;
  int *ilist;
  int *jlist;
  int *numneigh, **firstneigh;
  int *runnerNumNeigh, *runnerFirstNeighbor, *runnerJList, *runnerTypes;

  ev_init(eflag, vflag); // initializes flags, which signal if energy and virial need to be tallied.

  double **x = atom->x;
  double **f = atom->f;
  int nlocal = atom->nlocal;
  int nghost = atom->nghost;
  int ntotal = nlocal + nghost;
  int *type = atom->type;
  tagint *tag = atom->tag; // We currently don't need those

  // Interface variables
  double *runnerLocalE, *runnerForce, *runnerLocalStress, *runnerStress, runnerEnergy, *lattice;
  runnerLocalE = new double[ntotal];
  runnerForce = new double[ntotal * 3];
  runnerLocalStress = new double[ntotal * 9];
  runnerStress = new double[9];
  lattice = new double[9];

  /*
  //////////////////////////// MPI TESTGROUND  /////////////////////////////
  int size, rank, natoms;

  MPI_Comm_size(world, &size);
  MPI_Comm_rank(world, &rank);

  // DON'T FORGET DEALLOCATION WHEN USING NEW !!!
  int *nlocal_mpi = new int[size];
  int *nglobal_mpi = new int[size];

  for (i = 0; i < size; i++) nlocal_mpi[i] = 0;
  natoms = 0;

  inum = list->inum;
  ilist = list->ilist;
  nlocal_mpi[rank] = inum;

  MPI_Allreduce(nlocal_mpi, nglobal_mpi, size, MPI_INT, MPI_SUM, world);
  MPI_Allreduce(&nlocal, &natoms, 1, MPI_INT, MPI_SUM, world);

  if (rank == 0)
  {
    for (i = 0; i < size; i++)
    {
      std::cout << nglobal_mpi[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "total number of atoms " << natoms << std::endl;
  }

  int start, end;
  start = 0;
  for (i = 0; i < rank; i++) start += nglobal_mpi[i] * 3;
  end = start + nlocal * 3;

  std::cout << "start " << start << " end " << end << " on "<< rank << std::endl;

  MPI_Barrier(world);

  double *xyz_local = new double[natoms * 3];
  double *xyz_global = new double[natoms * 3];

  for (i = 0; i < natoms * 3; i++) xyz_local[i] = 0;
  for (i = 0; i < natoms * 3; i++) xyz_global[i] = 0;

  double xtmp, ytmp, ztmp;
  for (ii = 0; ii < inum; ii++)
  {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    xyz_local[start] = xtmp;
    xyz_local[start+1] = ytmp;
    xyz_local[start+2] = ztmp;
    start += 3;
  }

  MPI_Reduce(xyz_local, xyz_global, natoms * 3, MPI_DOUBLE, MPI_SUM, 0, world);

  if (rank == 0)
  {
    ii = 0;
    for (i = 0; i < natoms; i++)
    {
      std::cout << "ATOM " << i <<" ";
      for (j = 0; j < 3; j++)
      {
        std::cout << xyz_global[ii] << " " ;
        ii++;
      }
    std::cout << std::endl;;
    }
  }

  delete [] nlocal_mpi;
  delete [] nglobal_mpi;
  delete [] xyz_local;
  delete [] xyz_global;
  //////////////////////////// MPI TESTGROUND  /////////////////////////////
  */

  if (debug) std::cout << "Entered PairRUNNER::compute" << std::endl;

  // allocate additional per-atom arrays
  if (atom->nmax > nmax) {
    memory->destroy(atCharge);
    memory->destroy(hirshVolume);
    memory->destroy(elecNegativity);
    nmax = atom->nmax;

    memory->create(atCharge,nmax,"pair:atCharge");
    memory->create(hirshVolume,nmax,"pair:hirshVolume");
    memory->create(elecNegativity,nmax,"pair:elecNegativity");
  }

  // Set additional per-atom arrays to zero
  for (i = 0; i < nmax; i++)
  {
    atCharge[i] = 0.0;
    hirshVolume[i] = 0.0;
    elecNegativity[i] = 0.0;
  }

  // Neighborlist information
  inum = list->inum; // number of local atoms
  ilist = list->ilist; // local index of local atoms
  numneigh = list->numneigh; // number of neighbors of local atoms (dimension inum)
  firstneigh = list->firstneigh; // pointer array to first neighbors of local atoms (dimension inum)

  // Calculate total number of neighbors
  int numneighSum = 0;
  for (ii = 0; ii < inum; ii++) numneighSum += numneigh[ii];

  // create  arrays for interface
  runnerNumNeigh = new int[inum];
  runnerFirstNeighbor = new int[inum];
  runnerJList = new int[numneighSum];
  runnerTypes = new int[ntotal];

  // Collect neighbor data
  int irunner = 0;
  for (ii = 0; ii < inum; ii++)
  {
    i = ilist[ii]; // get local index of atom ii
    jlist = firstneigh[i]; // pointer to first neighbor of atom
    jnum = numneigh[i]; // number of neighbors jj of atom
    runnerNumNeigh[ii] = jnum;

    for (jj = 0; jj < jnum; jj++)
    {
      j = jlist[jj]; // index of neighbor jj
      j &= NEIGHMASK; // masks bits, which encode pair information
      if (jj == 0) runnerFirstNeighbor[ii] = irunner + 1; // plus one due to Fortran
      runnerJList[irunner] = j;
      irunner++; // runs till numNeighSum
    }
  }
  // collect atomic numbers of all atoms by converting element id to atomic number using map
  for (ii = 0; ii < ntotal; ii++) runnerTypes[ii] = map[type[ii]];

  // get lattice parameters
  lattice[0] = domain->xprd;
  lattice[1] = 0.0;
  lattice[2] = 0.0;
  lattice[3] = domain->xy;
  lattice[4] = domain->yprd;
  lattice[5] = 0.0;
  lattice[6] = domain->xz;
  lattice[7] = domain->yz;
  lattice[8] = domain->zprd;

  if (debug) std::cout << "Transfer atoms and neighbor lists to RuNNer interface" << std::endl;

  runner_lammps_interface_transfer_atoms_and_neighbor_lists(&nlocal, &nghost, runnerTypes, &inum,
    &numneighSum, ilist, runnerNumNeigh, runnerFirstNeighbor, runnerJList, lattice, &x[0][0], &lperiodic);

  if (debug) std::cout << "RuNNer short-range predicition" << std::endl;

  runner_lammps_interface_short_range(&nlocal, &nghost, &inum, ilist,
    &runnerEnergy, runnerLocalE, runnerStress, runnerLocalStress,
    runnerForce, hirshVolume, atCharge, elecNegativity);

  if (lHirshVolume)
  {
    if (debug) std::cout << "RuNNer long-range vdW interactions" << std::endl;

    // communicate Hirshfeld volumes from local atoms to ghost atoms
    commstyle = COMMHIRSHVOLUME;
    comm->forward_comm(this);

    // calculate dispersion energies and forces using Hirshfeld volumes and
    // volume gradients (stored on runner side)
    runner_lammps_interface_hirshfeld_vdw(&nlocal, &nghost, &inum, ilist,
      hirshVolume, &runnerEnergy, runnerForce);
  }

   /*
  Copy results from RuNNer back into LAMMPS atom array
  */

  // Forces
  irunner = 0;
  for (ii = 0; ii < ntotal; ii++)
  {
    for (jj = 0; jj < 3; jj++)
    {
      f[ii][jj] += runnerForce[irunner]; // runnerForce is a vector
      irunner++;
    }
  }

  // Potential energy
  if (eflag_global) eng_vdwl = runnerEnergy;

  // Local energy
  if (eflag_atom) for (ii = 0; ii < ntotal; ii++) eatom[ii] = runnerLocalE[ii];

  // Stress
  if (vflag_global)
  {
    virial[0] = runnerStress[0];
    virial[1] = runnerStress[4];
    virial[2] = runnerStress[8];
    virial[3] = (runnerStress[3] + runnerStress[1]) * 0.5;
    virial[4] = (runnerStress[2] + runnerStress[6]) * 0.5;
    virial[5] = (runnerStress[5] + runnerStress[7]) * 0.5;
  }

  // Local stress
  if (vflag_atom)
  {
    int iatom = 0;
    for (ii = 0; ii < ntotal; ii++) {
      vatom[ii][0] += runnerLocalStress[iatom + 0];
      vatom[ii][1] += runnerLocalStress[iatom + 4];
      vatom[ii][2] += runnerLocalStress[iatom + 8];
      vatom[ii][3] += (runnerLocalStress[iatom + 3] + runnerLocalStress[iatom + 1]) * 0.5;
      vatom[ii][4] += (runnerLocalStress[iatom + 2] + runnerLocalStress[iatom + 6]) * 0.5;
      vatom[ii][5] += (runnerLocalStress[iatom + 5] + runnerLocalStress[iatom + 7]) * 0.5;
      iatom += 9;
    }
  }

  // Deallocate internal arrays
  delete[] runnerTypes;
  delete[] runnerNumNeigh;
  delete[] runnerFirstNeighbor;
  delete[] runnerJList;
  delete[] runnerLocalE;
  delete[] runnerForce;
  delete[] runnerStress;
  delete[] runnerLocalStress;
  delete[] lattice;
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */
void PairRUNNER::settings(int narg, char ** /* arg */)
{
  if (narg != 0) error->all(FLERR, "Illegal pair_style command");

  // check if linked to the correct RUNNER library API version
  if (runner_lammps_api_version() != 1)
    error->all(FLERR,
               "RUNNER LAMMPS wrapper API version is not compatible "
               "with this version of LAMMPS");
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */
void PairRUNNER::allocate()
{
  allocated = 1; // sets allocated flag, checked in coeff function
  int np1 = atom->ntypes + 1; // +1 because typeID start at 1 not 0

  setflag = memory->create(setflag, np1, np1, "pair:setflag");
  cutsq = memory->create(cutsq, np1, np1, "pair:cutsq");
  map = new int[np1];
}

void PairRUNNER::coeff(int narg, char **arg)
{
  if (!allocated) allocate();

  int ntypes = atom->ntypes;

  // We only have the mapping of the element to type in the pair_coeff line
  // narg 0 and 1 are two wildcards * * for I,J, so mapping declaration starts at 2
  if (narg != (2 + ntypes))
    error->all(FLERR, "Number of arguments {} is not correct, it should be {}", narg, 2 + ntypes);

  // iarg - 1, because we start at 2, so first entry is 1.
  for (int iarg = 2; iarg < narg; iarg++) map[iarg - 1] = utils::inumeric(FLERR, arg[iarg], false, lmp);

  // clear setflag since coeff() might be called once with I,J = * *
  for (int iat = 1; iat <= ntypes; iat++)
    for (int jat = iat; jat <= ntypes; jat++) setflag[iat][jat] = 0;

  // set setflag i,j for type pairs where both are mapped to elements
  int count = 0;
  for (int iat = 1; iat <= ntypes; iat++)
    for (int jat = iat; jat <= ntypes; jat++)
      if (map[iat] >= 0 && map[jat] >= 0)
      {
        setflag[iat][jat] = 1;
        count++;
      }

  if (count == 0) error->all(FLERR, "Incorrect args for pair coefficients");

  // Read model coefficients and do initialization on RuNNer side.
  // Returns max cutoff for LAMMPS neighborlist.
  // Also returns booleans if additional atomic properties are predicted,
  // which need to be communicated between local and ghost atoms.
  std::string finputnn = "input.nn";
  int n_input_nn_len = strlen(finputnn.c_str());
  runner_lammps_interface_init(finputnn.c_str(),&n_input_nn_len, &cutoff,
    &lAtCharge, &lElecNegativity, &lHirshVolume);
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */
void PairRUNNER::init_style()
{
  // Require newton pair on
  // Switches reverse communication on on every time step, which adds data
  // from ghost atoms to corresponding local atoms
  // Required for RuNNer forces
  if (force->newton_pair != 1) error->all(FLERR, "Pair style runner requires newton pair on");

  // request full neighbor list
  neighbor->add_request(this, NeighConst::REQ_FULL);
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */
double PairRUNNER::init_one(int /*i*/, int /*j*/)
{
  // This function is called in the init phase of the simulation
  // It returns the cutoff, which is then used by LAMMPS for the neighborlist calculation
  return cutoff;
}

/* ----------------------------------------------------------------------
   communication between local and ghost atoms
------------------------------------------------------------------------- */

// pack local atom information into communication buffer.
int PairRUNNER::pack_forward_comm(int n, int *list, double *buf, int pbc_flag, int *pbc)
{
  int i,j,m;

  if (commstyle == COMMHIRSHVOLUME)
  {
    m = 0;
    for (i = 0; i < n; i++)
    {
      j = list[i];
      buf[m++] = hirshVolume[j];
    }
  }
  else if (commstyle == COMMATCHARGE)
  {
    m = 0;
    for (i = 0; i < n; i++)
    {
      j = list[i];
      buf[m++] = atCharge[j];
    }
  }
  else if (commstyle == COMMELECNEGATIVITY)
  {
    m = 0;
    for (i = 0; i < n; i++)
    {
      j = list[i];
      buf[m++] = elecNegativity[j];
    }
  }

  return m;
}

// unpack local atom information from buffer into ghost atom storage.
void PairRUNNER::unpack_forward_comm(int n, int first, double *buf)
{
  int i,m,last;

  if (commstyle == COMMHIRSHVOLUME)
  {
    m = 0;
    last = first + n;
    for (i = first; i < last; i++) hirshVolume[i] = buf[m++];
  }
  else if (commstyle == COMMATCHARGE)
  {
    m = 0;
    last = first + n;
    for (i = first; i < last; i++) atCharge[i] = buf[m++];
  }
  else if (commstyle == COMMELECNEGATIVITY)
  {
    m = 0;
    last = first + n;
    for (i = first; i < last; i++) elecNegativity[i] = buf[m++];
  }
}

// pack ghost atom contributions into communication buffer.
int PairRUNNER::pack_reverse_comm(int n, int first, double *buf)
{
  int i,m,last;

  if (commstyle == COMMHIRSHVOLUME)
  {
    m = 0;
    last = first + n;
    for (i = first; i < last; i++) buf[m++] = hirshVolume[i];
  }
  else if (commstyle == COMMATCHARGE)
  {
    m = 0;
    last = first + n;
    for (i = first; i < last; i++) buf[m++] = atCharge[i];
  }
  else if (commstyle == COMMELECNEGATIVITY)
  {
    m = 0;
    last = first + n;
    for (i = first; i < last; i++) buf[m++] = elecNegativity[i];
  }
  return m;
}

// Add ghost atom contributions from communication buffer to local atom.
void PairRUNNER::unpack_reverse_comm(int n, int *list, double *buf)
{
  int i,j,m;

  if (commstyle == COMMHIRSHVOLUME)
  {
    m = 0;
    for (i = 0; i < n; i++)
    {
      j = list[i];
      hirshVolume[j] += buf[m++];
    }
  }
  else if (commstyle == COMMATCHARGE)
  {
    m = 0;
    for (i = 0; i < n; i++)
    {
      j = list[i];
      atCharge[j] += buf[m++];
    }
  }
  else if (commstyle == COMMELECNEGATIVITY)
  {
    m = 0;
    for (i = 0; i < n; i++)
    {
      j = list[i];
      elecNegativity[j] += buf[m++];
    }
  }
}
