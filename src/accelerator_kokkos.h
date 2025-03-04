/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifndef LMP_ACCELERATOR_KOKKOS_H
#define LMP_ACCELERATOR_KOKKOS_H

// true interface to KOKKOS
// used when KOKKOS is installed

#ifdef LMP_KOKKOS

#include "kokkos.h"
#include "atom_kokkos.h"
#include "comm_kokkos.h"
#include "comm_tiled_kokkos.h"
#include "domain_kokkos.h"
#include "neighbor_kokkos.h"
#include "memory_kokkos.h"
#include "modify_kokkos.h"

#define LAMMPS_INLINE KOKKOS_INLINE_FUNCTION

#else

// dummy interface to KOKKOS
// needed for compiling when KOKKOS is not installed

#include "atom.h"
#include "comm_brick.h"
#include "comm_tiled.h"
#include "domain.h"
#include "neighbor.h"
#include "memory.h"
#include "modify.h"

#define LAMMPS_INLINE inline

namespace LAMMPS_NS {

class KokkosLMP {
 public:
  int kokkos_exists;
  int nthreads;
  int ngpus;
  int numa;

  KokkosLMP(class LAMMPS *, int, char **) {kokkos_exists = 0;}
  ~KokkosLMP() {}
  void accelerator(int, char **) {}
  int neigh_list_kokkos(int) {return 0;}
  int neigh_count(int) {return 0;}
};

class AtomKokkos : public Atom {
 public:
  tagint **k_special;
  AtomKokkos(class LAMMPS *lmp) : Atom(lmp) {}
  ~AtomKokkos() {}
  void sync(const ExecutionSpace /*space*/, unsigned int /*mask*/) {}
  void modified(const ExecutionSpace /*space*/, unsigned int /*mask*/) {}
};

class CommKokkos : public CommBrick {
 public:
  CommKokkos(class LAMMPS *lmp) : CommBrick(lmp) {}
  ~CommKokkos() {}
};

class CommTiledKokkos : public CommTiled {
 public:
  CommTiledKokkos(class LAMMPS *lmp) : CommTiled(lmp) {}
  CommTiledKokkos(class LAMMPS *lmp, Comm *oldcomm) : CommTiled(lmp,oldcomm) {}
  ~CommTiledKokkos() {}
};

class DomainKokkos : public Domain {
 public:
  DomainKokkos(class LAMMPS *lmp) : Domain(lmp) {}
  ~DomainKokkos() {}
};

class NeighborKokkos : public Neighbor {
 public:
  NeighborKokkos(class LAMMPS *lmp) : Neighbor(lmp) {}
  ~NeighborKokkos() {}
};

class MemoryKokkos : public Memory {
 public:
  MemoryKokkos(class LAMMPS *lmp) : Memory(lmp) {}
  ~MemoryKokkos() {}
  void grow_kokkos(tagint **, tagint **, int, int, const char*) {}
};

class ModifyKokkos : public Modify {
 public:
  ModifyKokkos(class LAMMPS *lmp) : Modify(lmp) {}
  ~ModifyKokkos() {}
};

class DAT {
 public:
  typedef double tdual_xfloat_1d;
  typedef double tdual_FFT_SCALAR_1d;
  typedef int tdual_int_1d;
  typedef int tdual_int_2d;
};

}

#endif
#endif
