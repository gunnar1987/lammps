/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(runner,PairRUNNER);
// clang-format on
#else

#ifndef LMP_PAIR_RUNNER_H
#define LMP_PAIR_RUNNER_H

#include "pair.h"

extern "C" {
int runner_lammps_api_version();
void runner_lammps_wrapper_init(const char *, int *, double *);
void runner_lammps_wrapper(int *, int *, int *, int *, int *, int *, int *, int *, int *, int *, double *,
                           double *, double *, double *, double *, double *, double *);
}

namespace LAMMPS_NS {

class PairRUNNER : public Pair {
 public:
  PairRUNNER(class LAMMPS *);
  ~PairRUNNER() override;

  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  void init_style() override;
  double init_one(int, int) override;
  void allocate();

 private:
  double cutoff;
  int *map;           // mapping from atom types to elements
};

}    // namespace LAMMPS_NS

#endif
#endif
