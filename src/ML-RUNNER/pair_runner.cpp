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
   Contributing authors: Knut Nikolas Lausch, Gunnar Schmitz

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
  unit_convert_flag = 0; // Currently no unit conversion
  no_virial_fdotr_compute = 1; // We calculate the virial ourselves and do not need to call virial_fdotr()
  map = nullptr;

  // additional per-atom arrays for communication
  nmax = 0;
  atCharge = nullptr;
  hirshVolume = nullptr;
  elecNegativity = nullptr;
  dEdQ = nullptr;
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
    memory->destroy(dEdQ);
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
  double *runnerLocalE, *runnerForce, *runnerLocalVirial, *runnerVirial, runnerEnergy, *lattice;
  runnerLocalE = new double[ntotal];
  runnerForce = new double[ntotal * 3];
  runnerLocalVirial = new double[ntotal * 9];
  runnerVirial = new double[9];
  lattice = new double[9];

  // MPI
  int size, rank;
  rank = comm->me;
  size = comm->nprocs;

  if (debug) std::cout << "Entered PairRUNNER::compute" << std::endl;

  // allocate additional per-atom arrays
  if (atom->nmax > nmax) {
    memory->destroy(atCharge);
    memory->destroy(hirshVolume);
    memory->destroy(elecNegativity);
    memory->destroy(dEdQ);
    nmax = atom->nmax;

    memory->create(atCharge,nmax,"pair:atCharge");
    memory->create(hirshVolume,nmax,"pair:hirshVolume");
    memory->create(elecNegativity,nmax,"pair:elecNegativity");
    memory->create(dEdQ, nmax, "pair:dEdQ");
  }

  // Set additional per-atom arrays to zero
  for (i = 0; i < nmax; i++)
  {
    atCharge[i] = 0.0;
    hirshVolume[i] = 0.0;
    elecNegativity[i] = 0.0;
    dEdQ[i] = 0.0;
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
    &runnerEnergy, runnerLocalE, runnerVirial, runnerLocalVirial,
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
      hirshVolume, &runnerEnergy, runnerForce, runnerVirial, runnerLocalVirial);
  }

  if (lAtCharge)
  {
    if (debug) std::cout << "RuNNer long-range electrostatics" << std::endl;

    // long-range electrostatics variables
    double runnerElecEnergy, *runnerElecForce;
    double *elecForceGlobal;
    double *dEdQGlobal;
    double *xyzGlobal, *qGlobal;
    int *zGlobal;
    int nAtoms;

    // pack electrostatics into one global structure
    nAtoms = pack_electrostatics(rank, size, inum, ilist, x, atCharge, runnerTypes, xyzGlobal, qGlobal, zGlobal);
    elecForceGlobal = new double[nAtoms * 3];
    dEdQGlobal = new double[nAtoms];

    if (rank == 0)
    {
      // calculate long-range electrostatics on root using global structure
      runner_lammps_interface_electrostatics(&nAtoms, &xyzGlobal[0], &zGlobal[0], lattice,
        &qGlobal[0], &runnerElecEnergy, &elecForceGlobal[0], &dEdQGlobal[0]);
    }

    MPI_Barrier(world);

    // Broadcast and unpack electrostatic results
    runnerElecForce = new double[ntotal * 3];
    unpack_electrostatics(rank, size, inum, ilist, nAtoms, ntotal, runnerElecEnergy,
      elecForceGlobal, dEdQGlobal, runnerElecForce, dEdQ);

    // add electrostatics contributions to short-range part
    runner_lammps_interface_add_electrostatics(&nlocal, &nghost, &inum, ilist,
      &runnerElecEnergy, runnerElecForce, dEdQ, &runnerEnergy, runnerForce);

    delete[] elecForceGlobal;
    delete[] dEdQGlobal;
    delete[] runnerElecForce;
    delete[] xyzGlobal; // allocated in pack electrostatics
    delete[] qGlobal; // allocated in pack electrostatics
    delete[] zGlobal; // allocated in pack electrostatics
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
    virial[0] = runnerVirial[0];
    virial[1] = runnerVirial[4];
    virial[2] = runnerVirial[8];
    virial[3] = runnerVirial[0];
    virial[4] = runnerVirial[6];
    virial[5] = runnerVirial[7];
  }

  // Local stress
  if (vflag_atom)
  {
    int iatom = 0;
    for (ii = 0; ii < ntotal; ii++) {
      vatom[ii][0] += runnerLocalVirial[iatom + 0];
      vatom[ii][1] += runnerLocalVirial[iatom + 4];
      vatom[ii][2] += runnerLocalVirial[iatom + 8];
      vatom[ii][3] += runnerLocalVirial[iatom + 0];
      vatom[ii][4] += runnerLocalVirial[iatom + 6];
      vatom[ii][5] += runnerLocalVirial[iatom + 7];
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
  delete[] runnerVirial;
  delete[] runnerLocalVirial;
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
  else if (commstyle == COMMDEDQ)
  {
    m = 0;
    for (i = 0; i < n; i++)
    {
      j = list[i];
      buf[m++] = dEdQ[j];
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
  else if (commstyle == COMMDEDQ)
  {
    m = 0;
    last = first + n;
    for (i = first; i < last; i++) dEdQ[i] = buf[m++];
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
  else if (commstyle == COMMDEDQ)
  {
    m = 0;
    last = first + n;
    for (i = first; i < last; i++) buf[m++] = dEdQ[i];
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
  else if (commstyle == COMMDEDQ)
  {
    m = 0;
    for (i = 0; i < n; i++)
    {
      j = list[i];
      dEdQ[j] += buf[m++];
    }
  }
}

/* ----------------------------------------------------------------------
   communication between processes.
------------------------------------------------------------------------- */

// pack local atom information into one global structure on root for electrostatics calculation.
int PairRUNNER::pack_electrostatics(int rank, int size, int inum, int *ilist, double **x,
  double *atCharge, int *runnerTypes, double * &xyzGlobal, double * &qGlobal, int * &zGlobal)
{
  int i, ii;
  int start, end;
  int natoms;

  // Determine how many local atoms are on each process.
  int *nLocal = new int[size];
  int *nGlobal = new int[size];
  for (i = 0; i < size; i++) nLocal[i] = 0;
  natoms = 0;
  nLocal[rank] = inum;
  MPI_Allreduce(nLocal, nGlobal, size, MPI_INT, MPI_SUM, world);

  // And how many atoms there are in this structure
  MPI_Allreduce(&inum, &natoms, 1, MPI_INT, MPI_SUM, world);

  // Determine array element boundaries for communication of positions on each process.
  // xyz is a flat array with 3 * natoms elements.
  start = 0;
  for (i = 0; i < rank; i++) start += nGlobal[i] * 3;
  end = start + inum * 3;

  double *xyzLocal = new double[natoms * 3];
  xyzGlobal = new double[natoms * 3]; // function gets a reference to xyz. Needs to be deleted outside function!

  for (i = 0; i < natoms * 3; i++) xyzLocal[i] = 0;
  for (i = 0; i < natoms * 3; i++) xyzGlobal[i] = 0;

  double xtmp, ytmp, ztmp;
  for (ii = 0; ii < inum; ii++)
  {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    xyzLocal[start] = xtmp;
    xyzLocal[start+1] = ytmp;
    xyzLocal[start+2] = ztmp;
    start += 3;
  }

  MPI_Barrier(world);

  // Communicate local positions to root preocess.
  MPI_Reduce(xyzLocal, xyzGlobal, natoms * 3, MPI_DOUBLE, MPI_SUM, 0, world);

  // Communication of atomic charges q and atomic numbers z.
  double *qLocal = new double[natoms];
  int *zLocal = new int[natoms];
  qGlobal = new double[natoms]; // function gets a reference to q. Needs to be deleted outside function!
  zGlobal = new int[natoms]; // function gets a reference to z. Needs to be deleted outside function!

  for (i = 0; i < natoms; i++) qGlobal[i] = 0;
  for (i = 0; i < natoms; i++) qLocal[i] = 0;
  for (i = 0; i < natoms; i++) zGlobal[i] = 0;
  for (i = 0; i < natoms; i++) zLocal[i] = 0;

  // Determine array element boundaries on each process again, since this time
  // only natoms elements need to be communicated.
  start = 0;
  for (i = 0; i < rank; i++) start += nGlobal[i];
  end = start + inum;
  for (ii = 0; ii < inum; ii++)
  {
    i = ilist[ii];
    qLocal[start] = atCharge[i];
    zLocal[start] = runnerTypes[i];
    start += 1;
  }
  MPI_Barrier(world);

  // Communicate local charges to root process.
  MPI_Reduce(qLocal, qGlobal, natoms, MPI_DOUBLE, MPI_SUM, 0, world);
  // Communicate local atomic numbers to root process.
  MPI_Reduce(zLocal, zGlobal, natoms, MPI_INT, MPI_SUM, 0, world);

  // Deallocation of local arrays.
  delete [] nLocal;
  delete [] nGlobal;
  delete [] xyzLocal;
  delete [] qLocal;
  delete [] zLocal;

  return natoms;
}

// Broadcast electrostatic results of one global structure on root and unpack information into local atom arrays.
void PairRUNNER::unpack_electrostatics(int rank, int size, int inum, int *ilist, int nAtoms, int ntotal,
  double elecEnergy, double * &elecForceGlobal, double * &dEdQGlobal, double *elecForce, double *dEdQ)
{
  int i, ii;
  int start;

  // set arrays to zero
  for (i = 0; i < ntotal * 3; i++) elecForce[i] = 0;
  for (i = 0; i < ntotal; i++) dEdQ[i] = 0;

  // Determine how many local atoms are on each process.
  int *nLocal = new int[size];
  int *nGlobal = new int[size];
  for (i = 0; i < size; i++) nLocal[i] = 0;
  nLocal[rank] = inum;
  MPI_Allreduce(nLocal, nGlobal, size, MPI_INT, MPI_SUM, world);

  // Broadcast global electrostatic results
  MPI_Bcast(&elecEnergy, 1, MPI_DOUBLE, 0, world);
  MPI_Bcast(elecForceGlobal, nAtoms * 3, MPI_DOUBLE, 0, world);
  MPI_Bcast(dEdQGlobal, nAtoms, MPI_DOUBLE, 0, world);

  // Determine starting index for adding global de_dq to local arrays
  start = 0;
  for (i = 0; i < rank; i++) start += nGlobal[i];

  for (ii = 0; ii < inum; ii++)
  {
    i = ilist[ii];
    dEdQ[i] = dEdQGlobal[start];
    start++;
  }

  // Determine starting index for adding global elec_force to local arrays
  start = 0;
  for (i = 0; i < rank; i++) start += nGlobal[i] * 3;

  for (ii = 0; ii < inum; ii++)
  {
    i = ilist[ii] * 3;
    elecForce[i] = elecForceGlobal[start];
    elecForce[i+1] = elecForceGlobal[start+1];
    elecForce[i+2] = elecForceGlobal[start+2];
    start += 3;
  }

  // Deallocation of local arrays.
  delete [] nLocal;
  delete [] nGlobal;
}
